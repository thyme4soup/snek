import os
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import random
import math
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import keras
import numpy as np
import imageio
from snake import Snake
from dqn_structures import ExplorationExploitationScheduler
from dqn_structures import ReplayMemory
from dqn_structures import DQN
from dqn_structures import TargetNetworkUpdater


tf.reset_default_graph()

# Control parameters
TRAIN = True
N_TRACKED_STATES = 3
MAX_EPISODE_LENGTH = 5000        # Equivalent of 5 minutes of gameplay at 60 frames per second
EVAL_FREQUENCY = 10000           # Number of frames the agent sees between evaluations
EVAL_STEPS = 5000                # Number of frames for one evaluation
NETW_UPDATE_FREQ = 10000         # Number of chosen actions between updating the target network.
                                 # According to Mnih et al. 2015 this is measured in the number of
                                 # parameter updates (every four actions), however, in the
                                 # DeepMind code, it is clearly measured in the number
                                 # of actions the agent choses
DISCOUNT_FACTOR = 0.99           # gamma in the Bellman equation
REPLAY_MEMORY_START_SIZE = 50000 # Number of completely random actions,
                                 # before the agent starts learning
MAX_FRAMES = 500000              # Total number of frames the agent sees
MEMORY_SIZE = 250000             # Number of transitions stored in the replay memory
NO_OP_STEPS = 2                  # Number of 'NOOP' or 'FIRE' actions at the beginning of an
                                 # evaluation episode
UPDATE_FREQ = 4                  # Every four actions a gradient descend step is performed
HIDDEN = 1024                    # Number of filters in the final convolutional layer. The output
                                 # has the shape (1,1,1024) which is split into two streams. Both
                                 # the advantage stream and value stream have the shape
                                 # (1,1,512). This is slightly different from the original
                                 # implementation but tests I did with the environment Pong
                                 # have shown that this way the score increases more quickly
LEARNING_RATE = 0.00025          # Set to 0.00025 in Pong for quicker results.
                                 # Hessel et al. 2017 used 0.0000625
BS = 32                          # Batch size

PATH = "output/"                 # Gifs and checkpoints will be saved here
SUMMARIES = "summaries"          # logdir for tensorboard
RUNID = 'run_1'
os.makedirs(PATH, exist_ok=True)
os.makedirs(os.path.join(SUMMARIES, RUNID), exist_ok=True)
SUMM_WRITER = tf.summary.FileWriter(os.path.join(SUMMARIES, RUNID))


def learn(session, replay_memory, main_dqn, target_dqn, batch_size, gamma):
    """
    Args:
        session: A tensorflow sesson object
        replay_memory: A ReplayMemory object
        main_dqn: A DQN object
        target_dqn: A DQN object
        batch_size: Integer, Batch size
        gamma: Float, discount factor for the Bellman equation
    Returns:
        loss: The loss of the minibatch, for tensorboard
    Draws a minibatch from the replay memory, calculates the
    target Q-value that the prediction Q-value is regressed to.
    Then a parameter update is performed on the main DQN.
    """
    # Draw a minibatch from the replay memory
    states, actions, rewards, new_states, terminal_flags = replay_memory.get_minibatch()
    # The main network estimates which action is best (in the next
    # state s', new_states is passed!)
    # for every transition in the minibatch
    arg_q_max = session.run(main_dqn.best_action, feed_dict={main_dqn.input:new_states})
    # The target network estimates the Q-values (in the next state s', new_states is passed!)
    # for every transition in the minibatch
    q_vals = session.run(target_dqn.q_values, feed_dict={target_dqn.input:new_states})
    double_q = q_vals[range(batch_size), arg_q_max]
    # Bellman equation. Multiplication with (1-terminal_flags) makes sure that
    # if the game is over, targetQ=rewards
    target_q = rewards + (gamma*double_q * (1-terminal_flags))
    # Gradient descend step to update the parameters of the main network
    loss, _ = session.run([main_dqn.loss, main_dqn.update],
                          feed_dict={main_dqn.input:states,
                                     main_dqn.target_q:target_q,
                                     main_dqn.action:actions})
    return loss

def generate_gif(frame_number, frames_for_gif, reward, path):
    """
        Args:
            frame_number: Integer, determining the number of the current frame
            frames_for_gif: A sequence of (x, n, 1) frames of a snake game in grayscale
            reward: Integer, Total reward of the episode that es ouputted as a gif
            path: String, path where gif is saved
    """

    imageio.mimsave(f'{path}{"SNAKE_frame_{0}_reward_{1}.gif".format(frame_number, reward)}',
                    frames_for_gif, duration=1/30)

def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def do_action_get_reward(snake, action):
    # Cache old values
    point = snake.getPoint()
    old_score = snake.getScore()
    old_head = snake.getHead()

    # Action to new state
    terminal = not snake.takeAction(action)

    # Reward calculation
    new_score = snake.getScore()
    new_head = snake.getHead()
    if terminal:
        reward = -1
    elif new_score == old_score:
        reward = 0.005 if dist(new_head, point) < dist(old_head, point) else 0
    else:
        reward = new_score - old_score

    # reward, terminal, terminal_life_lost
    return reward, terminal, terminal

def clip_reward(reward):
    return reward
    """
    if reward > 0:
        return 1
    elif reward == 0:
        return 0
    else:
        return -1
    """

def train():
    """Contains the training and evaluation loops"""
    my_replay_memory = ReplayMemory(size=MEMORY_SIZE, batch_size=BS)   # (★)
    update_networks = TargetNetworkUpdater(MAIN_DQN_VARS, TARGET_DQN_VARS)

    explore_exploit_sched = ExplorationExploitationScheduler(
        snake.ACTION_SHAPE, replay_memory_start_size=REPLAY_MEMORY_START_SIZE,
        max_frames=MAX_FRAMES)

    with tf.Session() as sess:
        sess.run(init)

        frame_number = 0
        rewards = []
        loss_list = []

        while frame_number < MAX_FRAMES:

            ########################
            ####### Training #######
            ########################
            epoch_frame = 0
            while epoch_frame < EVAL_FREQUENCY:
                snake._fresh()
                terminal_life_lost = False
                episode_reward_sum = 0
                for _ in range(MAX_EPISODE_LENGTH):
                    # (4★)
                    history = snake.getStates()
                    while len(history) < N_TRACKED_STATES:
                        history.insert(0, history[0])
                    state = np.array(history[(-1 * N_TRACKED_STATES):])
                    reshaped = np.reshape(state, MAIN_DQN.shape)
                    planned = sess.run(MAIN_DQN.best_action, feed_dict={MAIN_DQN.input:[reshaped]})[0]
                    action = explore_exploit_sched.get_action(planned, frame_number)
                    # (5★)
                    reward, terminal, terminal_life_lost = do_action_get_reward(snake, action)
                    processed_new_frame = snake.getStates()[-1]

                    frame_number += 1
                    epoch_frame += 1
                    episode_reward_sum += reward

                    # Clip the reward
                    clipped_reward = clip_reward(reward)

                    # (7★) Store transition in the replay memory
                    my_replay_memory.add_experience(action=action,
                                                    frame=processed_new_frame[:, :],
                                                    reward=clipped_reward,
                                                    terminal=terminal_life_lost)

                    if frame_number % UPDATE_FREQ == 0 and frame_number > REPLAY_MEMORY_START_SIZE:
                        loss = learn(sess, my_replay_memory, MAIN_DQN, TARGET_DQN,
                                     BS, gamma = DISCOUNT_FACTOR) # (8★)
                        loss_list.append(loss)
                    if frame_number % NETW_UPDATE_FREQ == 0 and frame_number > REPLAY_MEMORY_START_SIZE:
                        update_networks(sess) # (9★)

                    if terminal:
                        terminal = False
                        break

                rewards.append(episode_reward_sum)

                # Output the progress:
                if len(rewards) % 10 == 0:
                    # Scalar summaries for tensorboard
                    if frame_number > REPLAY_MEMORY_START_SIZE:
                        summ = sess.run(PERFORMANCE_SUMMARIES,
                                        feed_dict={LOSS_PH:np.mean(loss_list),
                                                   REWARD_PH:np.mean(rewards[-100:])})

                        SUMM_WRITER.add_summary(summ, frame_number)
                        loss_list = []
                    # Histogramm summaries for tensorboard
                    summ_param = sess.run(PARAM_SUMMARIES)
                    SUMM_WRITER.add_summary(summ_param, frame_number)

                    print(len(rewards), frame_number, np.mean(rewards[-100:]))
                    with open('rewards.dat', 'a') as reward_file:
                        print(len(rewards), frame_number,
                              np.mean(rewards[-100:]), file=reward_file)

            ########################
            ###### Evaluation ######
            ########################
            terminal = True
            gif = True
            frames_for_gif = []
            eval_rewards = []
            evaluate_frame_number = 0

            for _ in range(EVAL_STEPS):
                if terminal:
                    snake._fresh()
                    episode_reward_sum = 0
                    terminal_life_lost = False
                    terminal = False

                history = snake.getStates()
                while len(history) < N_TRACKED_STATES:
                    history.insert(0, history[0])
                state = np.array(history[(-1 * N_TRACKED_STATES):])
                reshaped = np.reshape(state, MAIN_DQN.shape)
                planned = sess.run(MAIN_DQN.best_action, feed_dict={MAIN_DQN.input:[reshaped]})[0]
                action = explore_exploit_sched.get_action(planned, frame_number)
                # (5★)
                reward, terminal, terminal_life_lost = do_action_get_reward(snake, action)
                processed_new_frame = snake.getStates()[-1]

                evaluate_frame_number += 1
                episode_reward_sum += reward

                if gif:
                    frames_for_gif.append(processed_new_frame)
                if terminal:
                    eval_rewards.append(episode_reward_sum)
                    gif = False # Save only the first game of the evaluation as a gif

            print("Evaluation score:\n", np.mean(eval_rewards))
            try:
                generate_gif(frame_number, frames_for_gif, eval_rewards[0], PATH)
            except IndexError:
                print("No evaluation game finished")

            #Save the network parameters
            saver.save(sess, PATH+'/my_model', global_step=frame_number)
            frames_for_gif = []

            # Show the evaluation score in tensorboard
            summ = sess.run(EVAL_SCORE_SUMMARY, feed_dict={EVAL_SCORE_PH:np.mean(eval_rewards)})
            SUMM_WRITER.add_summary(summ, frame_number)
            with open('rewardsEval.dat', 'a') as eval_reward_file:
                print(frame_number, np.mean(eval_rewards), file=eval_reward_file)

snake = Snake()
# main DQN and target DQN networks:
with tf.variable_scope('mainDQN'):
    MAIN_DQN = DQN(snake.ACTION_SHAPE, snake.BOARD_SHAPE, HIDDEN, LEARNING_RATE, N_TRACKED_STATES)  # (★★)
with tf.variable_scope('targetDQN'):
    TARGET_DQN = DQN(snake.ACTION_SHAPE, snake.BOARD_SHAPE, HIDDEN, LEARNING_RATE, N_TRACKED_STATES)               # (★★)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

MAIN_DQN_VARS = tf.trainable_variables(scope='mainDQN')
TARGET_DQN_VARS = tf.trainable_variables(scope='targetDQN')

LAYER_IDS = ["conv1", "conv2", "conv3", "denseAdvantage",
             "denseAdvantageBias", "denseValue", "denseValueBias"]

# Scalar summaries for tensorboard: loss, average reward and evaluation score
with tf.name_scope('Performance'):
    LOSS_PH = tf.placeholder(tf.float32, shape=None, name='loss_summary')
    LOSS_SUMMARY = tf.summary.scalar('loss', LOSS_PH)
    REWARD_PH = tf.placeholder(tf.float32, shape=None, name='reward_summary')
    REWARD_SUMMARY = tf.summary.scalar('reward', REWARD_PH)
    EVAL_SCORE_PH = tf.placeholder(tf.float32, shape=None, name='evaluation_summary')
    EVAL_SCORE_SUMMARY = tf.summary.scalar('evaluation_score', EVAL_SCORE_PH)

PERFORMANCE_SUMMARIES = tf.summary.merge([LOSS_SUMMARY, REWARD_SUMMARY])

# Histogramm summaries for tensorboard: parameters
with tf.name_scope('Parameters'):
    ALL_PARAM_SUMMARIES = []
    for i, Id in enumerate(LAYER_IDS):
        with tf.name_scope('mainDQN/'):
            MAIN_DQN_KERNEL = tf.summary.histogram(Id, tf.reshape(MAIN_DQN_VARS[i], shape=[-1]))
        ALL_PARAM_SUMMARIES.extend([MAIN_DQN_KERNEL])
PARAM_SUMMARIES = tf.summary.merge(ALL_PARAM_SUMMARIES)

if TRAIN:
    train()
