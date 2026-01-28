"""
Edited by Jerry Jin: run MDP
"""

#import tensorflow as tf
import argparse
import os.path as osp
import sys
sys.path.append('../')
from simulator.envs import *
from tools.load_data import *
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
#from algo.base import SummaryObj

from tools.create_envs import *

base_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
log_dir = osp.join(base_dir, 'log')
data_dir = osp.join(base_dir, 'data')


def running_example(args, training_round=1400, fleet_help=False):

    log_dir = '../logs/' + 'synthetic/'+'greedy/'+'debug'
    mkdir_p(log_dir)
    writer=SummaryWriter(log_dir)

    random.seed(0)
    np.random.seed(0)

    if args.grid_num == 100:
        env, M, N, central_node_ids, _ = create_OD(fleet_help)
    elif args.grid_num == 121:
        env, args.M, args.N, _, args.grid_num=load_envs_DiDi121(driver_num=args.driver_num)
    elif args.grid_num == 143:
        env, args.M, args.N, _, args.grid_num=load_envs_NYU143( driver_num=args.driver_num)

    # initialize the model
    #config = tf.ConfigProto(log_device_placement=False)
    #config.gpu_options.allow_growth = True
    #sess = tf.Session(config=config)

    #summary = SummaryObj(log_dir=log_dir, log_name=algo, n_group=1, sess=sess)
    #summary.register(['KL', 'Entropy', 'Fleet-ORR', 'Fake-ORR', 'ORR', 'GMV'])
    
    env.reset_randomseed(0)

    for iteration in range(training_round):
        print('\n---- ROUND: #{} ----'.format(iteration))

        order_response_rates = []
        T = 0
        max_iter = 144

        states_node, states, order_list, order_idx, order_feature, global_order_states = env.reset()

        gmv = []
        fake_orr = []
        fleet_orr = []
        kl = []
        entropy = []

        grid_num=100

        while T < max_iter:
            '''
            global_order_states= [order_id, begin id, end id, price, duration]
            order_id_pairs = [ (node id , order id),(),() ]* valid nodes 
            orders = [ order class * K]*valid nodes
            '''
            action_ids=[]
            for node_id in range(grid_num):
                feature=torch.Tensor(order_list[node_id])
                sort_logit,rank = torch.sort(feature[:,2],descending=True)
                action_ids.append([order_idx[node_id][id] for id in rank])

            orders = env.get_orders_by_id(action_ids)

            serve_drivers_ids, states_node, states, order_list, order_idx, order_feature, global_order_states = env.step(orders, generate_order=1)
            # distribution should gotten after step
            dist = env.step_get_distribution()
            entr_value = env.step_get_entropy()
            order_dist, driver_dist = dist[:, 0], dist[:, 1]
            kl_value = np.sum(order_dist * np.log(order_dist / driver_dist))
            kl.append(kl_value)
            entropy.append(entr_value)
            gmv.append(env.gmv)
            fake_orr.append(env.fake_response_rate)
            fleet_orr.append(env.fleet_response_rate)
            if env.order_response_rate >= 0:
                order_response_rates.append(env.order_response_rate)

            

            if T % 50 == 0:
                print(
                    'City_time: [{0:<5d}], Order_response_rate: [{1:<.4f}], KL: [{2:<.4f}], Entropy: [{3:<.4f}], Fake_orr: [{4:<.4f}], Fleet_arr: [{5:<.4f}], Idle_drivers: [{6}], Ori_order_num: [{7}], Fleet_drivers: [{8}]'.format(
                        env.city_time - 1, env.order_response_rate, kl_value, entr_value, env.fake_response_rate,
                        env.fleet_response_rate, env.ori_idle, env.ori_order_num, env.ori_fleet
                    ))

            T += 1
        print('>>> Mean_ORR: [{0:<.6f}] GMV: [{1}] Mean_KL: [{2}] Mean_Entropy: [{3}]'.format(
            np.mean(order_response_rates), np.sum(gmv), np.mean(kl), np.mean(entropy)))
        writer.add_scalar('train ORR',np.mean(order_response_rates),iteration)
        writer.add_scalar('train GMV',np.sum(gmv),iteration)
        writer.add_scalar('train KL',np.mean(kl),iteration)
        writer.add_scalar('train Entropy',np.mean(entropy),iteration)

        '''
        summary.write({
            'KL': np.mean(kl),
            'Entropy': np.mean(entropy),
            'Fake-ORR': np.mean(fake_orr),
            'Fleet-ORR': np.mean(fleet_orr),
            'ORR': np.mean(order_response_rates),
            'GMV': np.sum(gmv)
        }, iteration)
        '''
        # model.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--algo', type=str, default='MDP', help='Algorithm Type, choices: MDP')
    parser.add_argument('-t', '--train_round', type=int, help='Training round limit', default=1400)
    parser.add_argument('-f', '--fleet_help', type=bool, help='Trigger for fleet management', default=False)
    args = parser.parse_args()
    args.driver_num = 2000
    args.grid_num = 121
    running_example(args, args.train_round)
