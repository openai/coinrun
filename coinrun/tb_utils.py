import tensorflow as tf
from mpi4py import MPI
from coinrun.config import Config
import numpy as np

def clean_tb_dir():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        if tf.gfile.Exists(Config.TB_DIR):
            tf.gfile.DeleteRecursively(Config.TB_DIR) 
        tf.gfile.MakeDirs(Config.TB_DIR)

    comm.Barrier()

class TB_Writer(object):
    def __init__(self, sess):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        clean_tb_dir()

        tb_writer = tf.summary.FileWriter(Config.TB_DIR + '/' + Config.RUN_ID + '_' + str(rank), sess.graph)
        total_steps = [0]

        should_log = (rank == 0 or Config.LOG_ALL_MPI)

        if should_log:
            hyperparams = np.array(Config.get_arg_text())
            hyperparams_tensor = tf.constant(hyperparams)

            summary_op = tf.summary.text("hyperparameters info", hyperparams_tensor)
            summary = sess.run(summary_op)

            tb_writer.add_summary(summary)

        def add_summary(_merged, interval=1):
            if should_log:
                total_steps[0] += 1

                if total_steps[0] % interval == 0:
                    tb_writer.add_summary(_merged, total_steps[0])
                    tb_writer.flush()

        tuples = []

        def make_scalar_graph(name):
            scalar_ph = tf.placeholder(name='scalar_' + name, dtype=tf.float32)
            scalar_summary = tf.summary.scalar(name, scalar_ph)
            merged = tf.summary.merge([scalar_summary])
            tuples.append((scalar_ph, merged))

        name_dict = {}
        curr_name_idx = [0]

        def log_scalar(x, name, step=-1):
            if not name in name_dict:
                name_dict[name] = curr_name_idx[0]
                tf_name = (name + '_' + Config.RUN_ID) if curr_name_idx[0] == 0 else name
                make_scalar_graph(tf_name)
                curr_name_idx[0] += 1

            idx = name_dict[name]

            scalar_ph, merged = tuples[idx]
            
            if should_log:
                if step == -1:
                    step = total_steps[0]
                    total_steps[0] += 1

                _merged = sess.run(merged, {scalar_ph: x})

                tb_writer.add_summary(_merged, step)
                tb_writer.flush()
        
        self.add_summary = add_summary
        self.log_scalar = log_scalar
