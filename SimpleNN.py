import tensorflow as tf
import numpy as np
import math
import timeit

import DataReader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("dataset", 'fb15k', "Dataset, [fb15k|wn18]")
flags.DEFINE_boolean("simple", False, "Use simple projection (weighted plus) or matrix projection.")
flags.DEFINE_integer("topk", 1, "Hits@topk, default 1.")
flags.DEFINE_integer("batch", 2500, "mini batch size, default 2500.")
flags.DEFINE_integer('embed', 100, "embedding size, default 100.")
flags.DEFINE_integer('max_iter', 1000, 'max iteration, default 1000.')
flags.DEFINE_string("load", "", "load data from disk")
flags.DEFINE_float("e", 1e-8, "epsilon, default 1e-8.")
flags.DEFINE_float("beta1", 0.9, "beta1, default 0.9.")
flags.DEFINE_float("beta2", 0.999, "beta2, default 0.999.")
flags.DEFINE_float("lr", 0.001, "learning rate, default 0.001.")
flags.DEFINE_string("amie", "./fb15k_amie_rules.csv", "AMIE rule file, only contains Rule,Confidence,PCA.Confidence.")
flags.DEFINE_float("pca", 1.0, "PCA confidence threshold, default 1.0.")
flags.DEFINE_float("confidence", 0.8, "confidence threshold, default 0.8.")

FLAGS = flags.FLAGS


class SimpleNN:
    """ Basic ProjE model

    This model combines two entities using a compositional matrix or weights and then project the combined embedding
    onto each relation space.
    """
    __initialized = False
    __simple_projection = False
    __trainable = []

    def __init__(self, k_embeddings, n_rel, n_ent, prefix="", simple=False):
        """ Initialize neural network with given parameters.
        :param k_embeddings: The size of embeddings (vector representations).
        :param n_rel: Number of relations.
        :param n_ent: Number of entities.
        :param simple: Use simple weighted combination or matrix combination.
        :return: N/A
        """
        self.__k_embeddings = k_embeddings
        self.__n_rel = n_rel
        self.__n_ent = n_ent
        self.__simple_projection = simple

        bound = math.sqrt(6) / math.sqrt(k_embeddings)
        bound_proj = math.sqrt(6) / math.sqrt(k_embeddings * 2 + k_embeddings)
        bound_simple_proj = math.sqrt(6) / math.sqrt(k_embeddings * 2)
        bound_h3 = math.sqrt(6) / math.sqrt(k_embeddings)
        bound_bias = math.sqrt(6) / math.sqrt(n_rel)

        # Create embeddings
        with tf.device("/cpu:0"):
            self.__ent_embeddings = tf.get_variable(prefix + "ent_embeddings", [n_ent, k_embeddings],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                              maxval=bound,
                                                                                              seed=250))
            # relation embedding, also served as the relation projection layer
            self.__rel_embeddings = tf.get_variable(prefix + "rel_embeddings", [n_rel, k_embeddings],
                                                    initializer=tf.random_uniform_initializer(minval=-bound_h3,
                                                                                              maxval=bound_h3,
                                                                                              seed=255))

            if self.__simple_projection:
                # combination layer. This is a simple, weighted combination.
                self.__combination_layer = tf.get_variable(prefix + "nn_ent_combination_layer", [1, k_embeddings * 2],
                                                           initializer=tf.random_uniform_initializer(
                                                               minval=-bound_simple_proj, maxval=bound_simple_proj,
                                                               seed=233))
            else:
                # combination layer, this will combine two entities using an
                # `unknown operator` which defined by this layer.
                self.__combination_layer = tf.get_variable(prefix + "nn_ent_combination_layer",
                                                           [k_embeddings * 2, k_embeddings],
                                                           initializer=tf.random_uniform_initializer(minval=-bound_proj,
                                                                                                     maxval=bound_proj,
                                                                                                     seed=283))
            # bias of combination layer
            self.__comb_bias = tf.get_variable(prefix + "comb_bias", [k_embeddings],
                                               initializer=tf.random_uniform_initializer(minval=-bound, maxval=bound,
                                                                                         seed=863))

            # bias of relation projection layer
            self.__bias = tf.get_variable(prefix + "nn_bias", [n_rel],
                                          initializer=tf.random_uniform_initializer(minval=-bound_bias,
                                                                                    maxval=bound_bias, seed=9876))

            self.__trainable.append(self.__ent_embeddings)
            self.__trainable.append(self.__rel_embeddings)
            self.__trainable.append(self.__combination_layer)
            self.__trainable.append(self.__bias)
            self.__trainable.append(self.__comb_bias)

    def __call__(self, inputs, scope=None):
        """Run NN with given inputs. This function will only return the result of this NN,
         it will not modify any parameters.
        :param inputs: a tensor with shape [BATCH_SIZE, 2], BATCH_SIZE can be any positive integer.
                        A row [1, 2] in `inputs` equals to the id of two entities. This NN will convert
                        them into a concatenation of two entity embeddings,
                        [1, (HEAD_NODE_EMBEDDING, TAIL_NODE_EMBEDDING)].
        :param scope: If there is only one NN in the program then this could be omitted. Otherwise each NN should have
                        a unique scope to make sure they do not share the same hidden layers.
        :return: a [1, n_rel] tensor, before use the output one should use `tf.softmax` to squash the
                  output to a [1, n_rel] tensor which has a sum of 1.0 and the value of each cell lies in [0,1].
        """
        with tf.variable_scope(scope or type(self).__name__) as scp:
            # After first execution in which all hidden layer variables are created, we reuse all variables.
            if self.__initialized:
                scp.reuse_variables()

            # convert entity id into embeddings
            x = tf.reshape(tf.nn.embedding_lookup(self.__ent_embeddings, inputs), [-1, self.__k_embeddings * 2])

            # relation projection layer, this is also the output layer which transform the shape of
            # a tensor to [BATCH_SIZE, n_rel].
            rel_layer = tf.transpose(self.__rel_embeddings)

            if self.__simple_projection:  # weighted combination
                weighted_embedding = x * self.__combination_layer
                head_embedding, tail_embedding = tf.split(1, 2, weighted_embedding)
                y = tf.nn.bias_add(tf.matmul(tf.tanh(head_embedding + tail_embedding), rel_layer), self.__bias)
            else:  # matrix combination
                tmp1 = tf.nn.bias_add(tf.matmul(x, self.__combination_layer), self.__comb_bias)
                y = tf.nn.bias_add(tf.matmul(tf.tanh(tmp1), rel_layer), self.__bias)

        return y


def load_amie_rules(raw_data):
    """ Load association rules from the result of AMIE
    """
    rule_map = dict()

    with open(FLAGS.amie) as f:
        _ = f.readline()  # skip column titles
        for line in f:
            rule, confidence, pca_confidence = line.rstrip().split(',')
            confidence = float(confidence)
            pca_confidence = float(pca_confidence)

            # skip if this rule has a score lower than threshold
            if confidence < FLAGS.confidence or pca_confidence < FLAGS.pca:
                continue

            rules = [x.strip().split() for x in rule.split('=>')]
            assert len(rules) == 2

            # skip if this is not a simple len-1 rule
            if len(rules[0]) != 3 or len(rules[1]) != 3:
                continue

            # both are a->b relation
            if rules[0][0] == rules[1][0] and rules[0][2] == rules[1][2]:
                rule_id = raw_data.rel2id[rules[0][1]]
                if rule_id not in rule_map:
                    rule_map[rule_id] = set()
                rule_map[rule_id].add(raw_data.rel2id[rules[1][1]])
            else:  # this is a->b and b->a relation
                rule_id = raw_data.rel2id[rules[0][1]]
                if rule_id not in rule_map:
                    rule_map[rule_id] = set()
                rule_map[rule_id].add(-raw_data.rel2id[rules[1][1]])

    return rule_map


def gen_inputs(raw_data):
    """ Generate [[head, tail], ...] from raw input data
    """
    inputs = []
    for (head, tail, rel) in raw_data.train['path']:
        inputs.append([head, tail])
    return np.asarray(inputs)


def gen_targets(inputs, n_rel, raw_data, rule_map):
    """ Generate [[rel], ...] w.r.t. generated inputs
    """
    targets = np.zeros([len(inputs), n_rel], dtype=np.float32)
    raw_nominator = np.zeros([len(inputs), n_rel], dtype=np.float32)
    raw_denominator = np.zeros([len(inputs)], dtype=np.float32)

    ht_input_map = dict()  # find all edge ids with head and tail info
    association_rule_tasks = list()
    for i in range(0, len(inputs)):
        head, tail = inputs[i]
        if head not in ht_input_map:
            ht_input_map[head] = dict()
        if tail not in ht_input_map[head]:
            ht_input_map[head][tail] = list()

        ht_input_map[head][tail].append(i)

        for rel in raw_data.train['adj'][head][tail]:
            raw_nominator[i][rel] = len(raw_data.hlmap[head][rel].union(raw_data.tlmap[tail][rel]))
            raw_denominator[i] += raw_nominator[i][rel]

            if rel in rule_map:
                associated_rules = rule_map[rel]
                for associated_rule in associated_rules:
                    if associated_rule < 0:
                        association_rule_tasks.append([tail, head, rel, associated_rule])
                    else:
                        association_rule_tasks.append([head, tail, rel, associated_rule])

    print "find", len(association_rule_tasks), "potential association rule tasks"
    task_completed = 0

    for task in association_rule_tasks:
        head, tail, orig, rule = task
        try:
            for edge_id in ht_input_map[head][tail]:
                if raw_nominator[edge_id][abs(rule)] == 0:
                    try:
                        raw_nominator[edge_id][abs(rule)] += len(
                            raw_data.hlmap[head][abs(rule)].union(raw_data.tlmap[tail][abs(rule)]))
                    except KeyError:
                        if rule < 0:
                            raw_nominator[edge_id][abs(rule)] += len(
                                raw_data.hlmap[tail][abs(orig)].union(raw_data.tlmap[head][abs(orig)]))
                        else:
                            raw_nominator[edge_id][abs(rule)] += len(
                                raw_data.hlmap[head][abs(orig)].union(raw_data.tlmap[tail][abs(orig)]))

                    raw_denominator[edge_id] += raw_nominator[edge_id][abs(rule)]
                    task_completed += 1
            pass
        except KeyError:
            continue

    print task_completed, "tasks are completed."

    for i in range(len(inputs)):
        targets[i] = raw_nominator[i] / raw_denominator[i]

    # for i in range(0, len(inputs)):
    #     head, tail = inputs[i]
    #     nrel = sum(
    #         [len(raw_data.hlmap[head][l].union(raw_data.tlmap[tail][l])) for l in raw_data.train['adj'][head][tail]])
    #     test_sum = 0.
    #     for rel in raw_data.train['adj'][head][tail]:
    #         targets[i][rel] = float(len(raw_data.hlmap[head][rel].union(raw_data.tlmap[tail][rel]))) / float(nrel)
    #         test_sum += targets[i][rel]
    #     try:
    #         assert abs(test_sum - 1.) <= 1e-5
    #     except AssertionError:
    #         raise AssertionError("expect " + str(1.) + " actual " + str(test_sum))

    return np.asarray(targets)


def gen_filtered_rels(raw_data):
    """ Generate [[all rels], ...] w.r.t. generated inputs, i-th element contains all the relations
    connect i-th entity pair in inputs.
    """
    filtered_rels = []
    max_edges = 0
    for p in raw_data.test['path']:
        head, tail, rel = p
        filtered_rel = set()
        try:
            for r in raw_data.test['adj'][head][tail]:
                filtered_rel.add(r)
            for r in raw_data.train['adj'][head][tail]:
                filtered_rel.add(r)
            for r in raw_data.valid['adj'][head][tail]:
                filtered_rel.add(r)
        except KeyError:
            pass
        max_edges = max(max_edges, len(filtered_rel))
        filtered_rels.append(filtered_rel)

    print "max_n_edges", max_edges
    return np.asarray(filtered_rels)


def run(raw_data):
    """ Construct operators for training and evaluating SimpleNN model.
    """
    model = SimpleNN(FLAGS.embed, raw_data.rel_id_max + 1, raw_data.entity_id_max + 1, simple=FLAGS.simple)

    with tf.device("cpu:0"):
        ph_input = tf.placeholder(tf.int32, [None, 2])
        ph_target = tf.placeholder(tf.float32, [None, raw_data.rel_id_max + 1])

        y = model(ph_input)

        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(y, ph_target))

        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr, beta1=FLAGS.beta1, beta2=FLAGS.beta2,
                                           epsilon=FLAGS.e)

        grads = optimizer.compute_gradients(loss, tf.trainable_variables())
        op_train = optimizer.apply_gradients(grads)
        op_test = tf.nn.softmax(y)

    return model, ph_input, ph_target, loss, op_train, op_test


def run_eval(session, ph_input, op_test, raw_data, filtered_rels, k=1):
    """ Executes evaluation and returns accuracy score.
    """
    total = 0
    raw_hits = 0
    filtered_hits = 0

    inputs = raw_data.test['path'][:, 0:2]
    targets = raw_data.test['path'][:, 2]

    with tf.device("/cpu:0"):
        _, top_rels = tf.nn.top_k(op_test, 20 * k)
        top_rels = session.run(top_rels, {ph_input: inputs})

    for i in range(len(top_rels)):
        total += 1
        idx = 0
        filtered_idx = 0
        while filtered_idx < k and idx < len(top_rels[i]):
            if top_rels[i][idx] == targets[i]:
                filtered_hits += 1
                break
            elif top_rels[i][idx] not in filtered_rels[i]:
                filtered_idx += 1
            idx += 1

        raw_hits += targets[i] in top_rels[i][:k]

    return float(raw_hits) / float(total), float(filtered_hits) / float(total)


def main(_):
    raw_data = DataReader.MetaPathData()

    if FLAGS.dataset == 'fb15k':
        raw_data.load_data('./data/FB15k/')
    else:
        print "unknown dataset"
        exit(-1)

    model, ph_input, ph_target, loss, op_train, op_test = run(raw_data)

    inputs = gen_inputs(raw_data)
    rule_map = load_amie_rules(raw_data)
    targets = gen_targets(inputs, raw_data.rel_id_max + 1, raw_data, rule_map)
    filtered_rels = gen_filtered_rels(raw_data)

    best_raw_acc = 0.
    best_raw_iter = -1

    best_filtered_acc = 0.
    best_filtered_iter = -1

    if FLAGS.batch <= 0:
        FLAGS.batch = len(raw_data.train['path'])

    with tf.Session() as session:

        tf.initialize_all_variables().run()

        for it in range(FLAGS.max_iter):
            print "--- Iteration", it, "---"

            start_time = timeit.default_timer()

            new_order = range(0, len(inputs))
            np.random.shuffle(new_order)
            inputs = inputs[new_order, :]
            targets = targets[new_order, :]

            accu_loss = 0.

            start = 0
            while start < len(inputs):
                end = min(start + FLAGS.batch, len(inputs))
                l, _ = session.run([loss, op_train], {ph_input: inputs[start:end, :], ph_target: targets[start:end, :]})
                accu_loss += l
                start = end

            print "\n\tloss:", accu_loss, "cost:", timeit.default_timer() - start_time, "seconds.\n"

            raw_acc, filtered_acc = run_eval(session, ph_input, op_test, raw_data, filtered_rels, k=FLAGS.topk)

            print "\n\traw", raw_acc, "\t\tfiltered", filtered_acc, "\n"

            if raw_acc > best_raw_acc:
                best_raw_acc = raw_acc
                best_raw_iter = it

            if filtered_acc > best_filtered_acc:
                best_filtered_acc = filtered_acc
                best_filtered_iter = it

            print "\tbest\n\traw", best_raw_acc, "(", best_raw_iter, ")", \
                "\tfiltered", best_filtered_acc, "(", best_filtered_iter, ")\n"

            print "--------------------------"


if __name__ == '__main__':
    tf.app.run()
