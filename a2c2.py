"""
Using:
tensorflow 1.0
Sisheng Liang  2019
"""
import sys
import numpy as np
import tensorflow as tf
import environment
import time
import parameters
import slow_down_cdf
import matplotlib.pyplot as plt
from multiprocessing import Process


# Superparameters, globle parameters
OUTPUT_GRAPH = True  # If this is False, no graph in tensorboard
RENDER = False  # rendering wastes time
GAMMA = 0.9  # reward discount in TD error
LR_A = 0.001  # learning rate for actor
LR_C = 0.01  # learning rate for critic

pa = parameters.Parameters()
input_width = pa.network_input_width
input_height = pa.network_input_height
output_dim = pa.network_output_dim


class Actor(object):

    def __init__(self, sess,  input_height, input_width,output_dim,lr=0.001):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, shape=(input_height, input_width), name="state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.advantage= tf.placeholder(tf.float32, None, name="td_error")  # TD_error

        with tf.variable_scope('Actor'):
            # l1 = tf.layers.dense(
            #     inputs=self.s,
            #     units=input_width,    # number of hidden units
            #     activation=tf.nn.relu,
            #     kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
            #     bias_initializer=tf.constant_initializer(0.1),  # biases
            #     name='l1'
            # )
            self.l1 = tf.layers.dense(
                inputs=self.s,
                units=input_width,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=self.l1,
                units=output_dim,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        # with tf.variable_scope('exp_v'):
            # self.log_prob = tf.log(self.acts_prob[0, self.a])
            # self.exp_v = tf.reduce_mean(self.log_prob * self.td_error)  # advantage (TD_error) guided loss

            # self.sy_logits_na = self.acts_logits
            # sy_ac_na = tf.one_hot(tf.transpose(sy_ac_na), self.ac_dim)
            # sy_logprob_n = tf.nn.softmax_cross_entropy_with_logits_v2(
            #     labels=sy_ac_na, logits=self.sy_logits_na)



        # with tf.variable_scope('train'):
        #     self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self,a, s):
        #s = s[np.newaxis, :]
        feed_dict = {self.a: a, self.s:s}
        log_prob = self.sess.run(self.log_prob, feed_dict)
        #print("exp_v",exp_v)
        return log_prob


    def get_one_act_prob(self, state):
        act_prob = self.sess.run(self.acts_prob[0],feed_dict={self.s: state})    # this [0] reduce the dimension
        #print("act_prob_test",act_prob)
        return act_prob

    def get_log_prob(self, action):
        logprob =tf.log((tf.squeeze(self.acts_prob[0]))[action])
        return logprob
    def update(self, lr, loss):
        with tf.variable_scope('train1'):
            self.train_op = self.sess.run(tf.train.AdamOptimizer(lr).minimize(loss),feed_dict={lr:LR_A,loss:loss})
        return


class Critic(object):
    def __init__(self, sess, input_height, input_width,lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, shape=(input_height, input_width), name="state")
        self.v_ = tf.placeholder(tf.float32, None, "v_next")
        self.r = tf.placeholder(tf.float32, None, name='r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=input_width,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            l2 = tf.layers.dense(
                inputs=l1,
                units=input_width,  # number of hidden units
                activation=tf.nn.relu,  # None
                # have to be linear to make sure the convergence of actor.
                # But linear approximator seems hardly learns the correct Q.
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l2'
            )


            self.v = tf.layers.dense(
                inputs=l2,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )
            # self.v = tf.reduce_sum(v_out)

        with tf.variable_scope('squared_TD_error'):


            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self):
        #s = s[np.newaxis, :]
        #s_ = s_[np.newaxis, :]
        #v_ = self.sess.run(self.v, feed_dict={self.s: s_})   # value @ next state
        v = tf.squeeze(self.v[0])  # value @ next state
        # print("value next, value",v_,v)
        # print('v.shape:', self.v.get_shape(), self.s.get_shape(), self.v_)
        # td_error, _ = self.sess.run([self.td_error, self.train_op],
        #tf.squeeze(self.v[0])                               {self.s: s, self.v_: v_, self.r: r,self.v: v})
        #self.v[0]
        #print("value",v)
        #print("value v[0]",v[0])
        return v

    def update(self, lr, loss):
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(loss)
            self.sess.run(self.train_op,feed_dict={lr:lr,loss:loss})
        return


sess = tf.Session()
actor = Actor(sess,input_height, input_width,output_dim,lr=LR_A)
critic = Critic(sess,input_height, input_width,lr=LR_C)     # we need a good teacher, so the teacher should learn faster than the actor
sess.run(tf.global_variables_initializer())
# #
if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)



def a2c(env):


    all_lengths = []
    average_lengths = []
    all_rewards = []
    entropy_term = 0

    for episode in range(pa.num_epochs):
        log_probs = []
        values = []
        rewards = []
        # time.sleep(0.1)
        #env.render()
        env.reset()
        s = env.observe()
        print("state",s)

        for steps in range(pa.num_seq_per_batch):

            acts_prob = actor.get_one_act_prob(s)
            csprob_n = np.cumsum(acts_prob)
            action = (csprob_n > np.random.rand()).argmax()
            print("action from step",action)

            s_next, rew, done, info = env.step(action, repeat=True)
            value = critic.learn(s)
            log_prob = actor.get_log_prob(action)  # this s_next could be s, not sure
            print("log_prob",log_prob)
            rewards.append(rew)
            values.append(value)
            log_probs.append(log_prob)
            print("log_probs",log_probs)

            s = s_next


            if done or steps == pa.num_seq_per_batch - 1:
                Qval = critic.learn(s_next)
                all_rewards.append(np.sum(rewards))
                all_lengths.append(steps)
                average_lengths.append(np.mean(all_lengths[-50:]))
                if episode % 50 == 0:
                    sys.stdout.write("episode: {}, reward: {}, total length: {}, average length: {} \n".format(episode,
                                                                                                               np.sum(
                                                                                                                   rewards),
                                                                                                               steps,
                                                                                                               average_lengths[
                                                                                                                   -1]))
                break

        # compute Q values
        Qvals = np.zeros_like(values)
        for t in reversed(range(len(rewards))):
            Qval = rewards[t] + GAMMA * Qval
            Qvals[t] = Qval

        # update actor critic
        values = values
        advantage = Qvals - values
        actor_loss = -tf.reduce_mean(np.array(log_probs)*advantage)
        critic_loss = tf.square((advantage).mean())
        # ac_loss = actor_loss + 0 * critic_loss + 0.000 * entropy_term  # 1 , 0.001
        tf.summary.FileWriter("logs/", sess.graph)
        # train_op1 = tf.train.AdamOptimizer(LR_A).minimize(actor_loss)
        #
        # train_op2 = tf.train.AdamOptimizer(LR_C).minimize(critic_loss)

        critic.update(LR_C, critic_loss)
        actor.update(LR_A,actor_loss)

        # ac_optimizer.zero_grad()
        # ac_loss.backward()
        # ac_optimizer.step()

    # Plot results
    #smoothed_rewards = pd.Series.rolling(pd.Series(all_rewards), 10).mean()
    #smoothed_rewards = [elem for elem in smoothed_rewards]
    plt.plot(all_rewards)
    # plt.plot(smoothend_rewards)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

    plt.plot(all_lengths)
    plt.plot(average_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Episode length')
    plt.show()


if __name__ == "__main__":
    env = environment.Env(pa, render=False, repre='image', end='no_new_job')
    a2c(env)




