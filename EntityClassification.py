import tensorflow as tf
import numpy as np
import math
import timeit

import DataReader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("dataset", 'fb15k', "Dataset, [fb15k|wn18]")
flags.DEFINE_boolean("simple", True, "Use simple projection (weighted plus) or matrix projection.")
flags.DEFINE_integer("topk", 1, "relation Hits@topk, default 1.")
flags.DEFINE_integer("ent_topk", 10, "entity Hits@topk, default 10.")
flags.DEFINE_integer("batch", 500, "mini batch size, default 2500.")
flags.DEFINE_integer('embed', 100, "embedding size, default 100.")
flags.DEFINE_integer('max_iter', 1000, 'max iteration, default 1000.')
flags.DEFINE_string("load", "", "load data from disk")
flags.DEFINE_float("e", 1e-8, "epsilon, default 1e-8.")
flags.DEFINE_float("beta1", 0.9, "beta1, default 0.9.")
flags.DEFINE_float("beta2", 0.999, "beta2, default 0.999.")
flags.DEFINE_float("lr", 0.001, "learning rate, default 0.001.")
flags.DEFINE_string("amie", "./fb15k_amie_rules.csv", "AMIE rule file, only contains Rule,Confidence,PCA.Confidence.")
flags.DEFINE_float("pca", 1.0, "PCA confidence threshold, default 1.0.")
flags.DEFINE_float("confidence", 0.7, "confidence threshold, default 0.8.")
flags.DEFINE_float("entropy_weight", 0.5, "wrong class entropy weight")

FLAGS = flags.FLAGS


class EntityMC:
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
        bound_bias = math.sqrt(6) / math.sqrt(n_ent)
        bound_output = math.sqrt(6) / math.sqrt(k_embeddings + n_ent)

        # Create embeddings
        with tf.device("/cpu:0"):
            self.__ent_embeddings = tf.get_variable(prefix + "ent_embeddings", [n_ent, k_embeddings],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                              maxval=bound,
                                                                                              seed=250))

            self.__rel_embeddings = tf.get_variable(prefix + "rel_embeddings", [n_rel, k_embeddings],
                                                    initializer=tf.random_uniform_initializer(minval=-bound,
                                                                                              maxval=bound,
                                                                                              seed=255))

            if self.__simple_projection:
                # combination layer. This is a simple, weighted combination.
                self.__combination_layer = tf.get_variable(prefix + "nn_combination_layer", [1, k_embeddings * 2],
                                                           initializer=tf.random_uniform_initializer(
                                                               minval=-bound_simple_proj, maxval=bound_simple_proj,
                                                               seed=233))
            else:
                # combination layer, this will combine two entities using an
                # `unknown operator` which defined by this layer.
                self.__combination_layer = tf.get_variable(prefix + "nn_combination_layer",
                                                           [k_embeddings * 2, k_embeddings],
                                                           initializer=tf.random_uniform_initializer(minval=-bound_proj,
                                                                                                     maxval=bound_proj,
                                                                                                     seed=283))
            # bias of combination layer
            self.__comb_bias = tf.get_variable(prefix + "comb_bias", [k_embeddings],
                                               initializer=tf.random_uniform_initializer(minval=-bound, maxval=bound,
                                                                                         seed=863))

            # self.__output_layer = tf.get_variable(prefix + "output_layer", [k_embeddings, n_ent],
            #                                      initializer=tf.random_uniform_initializer(minval=-bound_output,
            #                                                                                maxval=bound_output,
            #                                                                                seed=79403))

            self.__output_bias = tf.get_variable(prefix + "nn_bias", [n_ent],
                                                 initializer=tf.random_uniform_initializer(minval=-bound_bias,
                                                                                           maxval=bound_bias,
                                                                                           seed=9876))

            self.__trainable.append(self.__ent_embeddings)
            self.__trainable.append(self.__rel_embeddings)
            self.__trainable.append(self.__combination_layer)
            # self.__trainable.append(self.__output_layer)
            self.__trainable.append(self.__output_bias)
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

            ent_inputs, rel_inputs = tf.split(1, 2, inputs)
            # convert entity id into embeddings
            x_ent = tf.nn.embedding_lookup(self.__ent_embeddings, ent_inputs)
            x_rel = tf.nn.embedding_lookup(self.__rel_embeddings, rel_inputs)

            x = tf.reshape(tf.concat(1, [x_ent, x_rel]), [-1, self.__k_embeddings * 2])

            if self.__simple_projection:
                weighted_embedding = x * self.__combination_layer
                head_embedding, rel_embedding = tf.split(1, 2, weighted_embedding)
                y = tf.nn.bias_add(
                    tf.matmul(tf.tanh(head_embedding + rel_embedding), tf.transpose(self.__ent_embeddings)),
                    self.__output_bias)
            else:
                tmp1 = tf.nn.bias_add(tf.matmul(x, self.__combination_layer), self.__comb_bias)
                y = tf.nn.bias_add(tf.matmul(tf.tanh(tmp1), tf.transpose(self.__ent_embeddings)), self.__output_bias)

        return y


def gen_inputs(raw_data):
    """ Generate [[head, rel], ...] from raw input data
    """
    inputs = []
    for (head, tail, rel) in raw_data.train['path']:
        inputs.append([head, rel])
    return np.asarray(inputs)


def gen_targets(inputs, raw_data):
    """ Generate [[tail_type], ...] w.r.t. generated inputs
    """
    targets = np.zeros([len(inputs), raw_data.entity_id_max + 1], dtype=np.float32)
    weights = np.zeros([len(inputs), raw_data.entity_id_max + 1], dtype=np.float32)
    weights[:, :] = FLAGS.entropy_weight

    for (i, edge) in enumerate(inputs):
        head, rel = edge

        for candidate in raw_data.hlmap[head][rel]:
            targets[i][candidate] = candidate / len(raw_data.hlmap[head][rel])
            weights[i][candidate] = 1.0

    return np.asarray(targets), np.asarray(weights)


def gen_filtered_classes(raw_data):
    filtered_classes = list()
    max_count = 0

    for i in range(len(raw_data.test['path'])):
        head, _, rel = raw_data.test['path'][i]

        filtered_class = set()

        try:
            tails = raw_data.hlmap[head][rel]
            for tail in tails:
                filtered_class.add(tail)
        except KeyError:
            pass

        try:
            tails = raw_data.hl_test_map[head][rel]
            for tail in tails:
                filtered_class.add(tail)
        except KeyError:
            pass

        try:
            tails = raw_data.hl_valid_map[head][rel]
            for tail in tails:
                filtered_class.add(tail)
        except KeyError:
            pass

        max_count = max(max_count, len(filtered_class))
        filtered_classes.append(filtered_class)

    return max_count, filtered_classes


def run(raw_data):
    """ Construct operators for training and evaluating SimpleNN model.
    """
    model = EntityMC(FLAGS.embed, raw_data.rel_id_max + 1, raw_data.entity_id_max + 1, simple=FLAGS.simple)

    with tf.device("cpu:0"):
        ph_input = tf.placeholder(tf.int32, [None, 2])
        ph_target = tf.placeholder(tf.float32, [None, raw_data.entity_id_max + 1])
        ph_weight = tf.placeholder(tf.float32, [None, raw_data.entity_id_max + 1])

        y = model(ph_input)

        loss = -tf.reduce_sum(ph_target * tf.log(tf.nn.softmax(y)) * ph_weight)
        # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, ph_target))

        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.lr, beta1=FLAGS.beta1, beta2=FLAGS.beta2,
                                           epsilon=FLAGS.e)

        grads = optimizer.compute_gradients(loss, tf.trainable_variables())
        op_train = optimizer.apply_gradients(grads)
        op_test = tf.nn.softmax(y)

    return model, ph_input, ph_target, ph_weight, loss, op_train, op_test


def run_eval(session, ph_input, op_test, raw_data, filtered_classes, k=1, max_offset=11):
    """ Executes evaluation and returns accuracy score.
    """
    total = 0
    raw_hits = 0
    filtered_hits = 0
    rank = 0
    filtered_rank = 0

    ent_inputs = raw_data.test['path'][:, 0]
    targets = raw_data.test['path'][:, 1]

    rel_inputs = raw_data.test['path'][:, 2]

    inputs = np.zeros([len(ent_inputs), 2], dtype=np.int32)
    inputs[:, 0] = ent_inputs
    inputs[:, 1] = rel_inputs

    with tf.device("/cpu:0"):
        _, top_rels = tf.nn.top_k(op_test, raw_data.entity_id_max + 1)
        # _, top_rels = tf.nn.top_k(op_test, max_offset + k)
        top_rels = session.run(top_rels, {ph_input: inputs})

    for i in range(len(top_rels)):
        total += 1
        idx = 0
        filtered_idx = 0

        while idx < len(top_rels[i]):
            if top_rels[i][idx] == targets[i]:
                rank += idx + 1
                filtered_rank += filtered_idx + 1
                break
            elif top_rels[i][idx] not in filtered_classes[i]:
                filtered_idx += 1
            idx += 1

        idx = 0
        filtered_idx = 0

        while filtered_idx < k and idx < len(top_rels[i]):
            if top_rels[i][idx] == targets[i]:
                filtered_hits += 1
                break
            elif top_rels[i][idx] not in filtered_classes[i]:
                filtered_idx += 1
            idx += 1

        raw_hits += targets[i] in top_rels[i][:k]

    return float(raw_hits) / float(total), float(filtered_hits) / float(total), float(rank) / float(total), float(
        filtered_rank) / float(total)


def main(_):

    print "simple:", FLAGS.simple

    raw_data = DataReader.MetaPathData()

    if FLAGS.dataset == 'fb15k':
        raw_data.load_data('./data/FB15k/')
    else:
        print "unknown dataset"
        exit(-1)

    inputs = gen_inputs(raw_data)
    targets, weights = gen_targets(inputs, raw_data)
    max_tail_offset, filtered_classes = gen_filtered_classes(raw_data)

    model, ph_input, ph_target, ph_weight, loss, op_train, op_test = run(raw_data)

    best_raw_acc = 0.
    best_raw_iter = -1

    best_filtered_acc = 0.
    best_filtered_iter = -1

    best_mean_rank = 99999
    best_mean_rank_iter = -1
    best_filtered_rank = 99999
    best_filtered_rank_iter = -1

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
            weights = weights[new_order, :]

            accu_loss = 0.

            start = 0
            while start < len(inputs):
                end = min(start + FLAGS.batch, len(inputs))
                tmp_weight = weights[start:end, :]
                tmp_weight = np.minimum(
                    tmp_weight + np.random.choice([0, 1], size=[len(tmp_weight), len(tmp_weight[0])], replace=True,
                                                  p=[0.995, 0.005]), 1)
                l, _ = session.run([loss, op_train], {ph_input: inputs[start:end, :],
                                                      ph_target: targets[start:end, :],
                                                      ph_weight: tmp_weight})
                accu_loss += l
                start = end
            print "\n\tloss:", accu_loss, "cost:", timeit.default_timer() - start_time, "seconds.\n"

            print "--- entity class prediction ---"

            raw_rel_acc, filtered_rel_acc, raw_rank, filtered_rank = run_eval(session, ph_input, op_test, raw_data, filtered_classes,
                                                     k=FLAGS.topk,
                                                     max_offset=max_tail_offset)

            print "\n\traw", raw_rel_acc, "\t\tfiltered", filtered_rel_acc, "\n"
            print "\n\traw", raw_rank, "\t\tfiltered", filtered_rank,"\n"

            if raw_rank < best_mean_rank:
                best_mean_rank = raw_rank
                best_mean_rank_iter = it

            if filtered_rank < best_filtered_rank:
                best_filtered_rank = filtered_rank
                best_filtered_rank_iter = it

            if raw_rel_acc > best_raw_acc:
                best_raw_acc = raw_rel_acc
                best_raw_iter = it

            if filtered_rel_acc > best_filtered_acc:
                best_filtered_acc = filtered_rel_acc
                best_filtered_iter = it

            print "\tbest\n\traw", best_raw_acc, "(", best_raw_iter, ")", \
                "\tfiltered", best_filtered_acc, "(", best_filtered_iter, ")\n"
            print "\tbest\n\traw", best_mean_rank, "(", best_mean_rank_iter, ")", \
                "\tfiltered", best_filtered_rank, "(", best_filtered_rank_iter, ")", "\n"

            print "--------------------------"


if __name__ == '__main__':
    tf.app.run()