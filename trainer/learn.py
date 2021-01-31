from agents.agent import NNAgent
from dynamic_models.n_linkage import kane
from gym_linkage.envs import LinkageEnv


# @hydra.main(config_path="../conf/config.yaml")
def main():

    M, F, params = kane(n=5)
    env = LinkageEnv(M, F, params, cfg.param_vals)
    # nn = get_model(states_size, cfg.train.nn_param, action_space.length)
    # agent = NNAgent(action_space, nn)
    # epsilon = cfg.train.epsilon
    # replay = []
    # data_collect = []
    # loss_log = []
    # total_steps = 0
    # for i in range(cfg.train.episodes):
    #     ep_steps = 0
    #     logger.info(f"Episode: {i}")
    #     ob, reward, done = env.reset(), 0, False
    #     while done is False:
    #         data = [i, *ob[:4]]
    #         data_collect.append(data)

    #         ep_steps += 1
    #         total_steps += 1
    #         action = agent.act(ob, reward, epsilon, done)
    #         new_ob, reward, done, _ = env.step(action)
    #         replay.append((ob, action, reward, new_ob))
    #         #print(ob, action, reward, new_ob )
    #         # use_replay_to_fit_model
    #         if total_steps > cfg.train.rm_size:
    #             if len(replay) >= cfg.train.buffer:
    #                 replay.pop(0)
    #             nn, loss_log = fit_model(
    #                 nn,
    #                 replay,
    #                 cfg.train.batch_size,
    #                 loss_log,
    #                 states_size,
    #                 gamma=cfg.train.gamma,
    #             )

    #         ob = new_ob
    #         epsilon = -1 / cfg.train.max_steps
    #         if done:
    #             break

    #     if total_steps % cfg.model.save_per_eps == 0:
    #         model_dir = os.path.join(cfg.model.dir, f"{cfg.id.count}")
    #         os.makedirs(model_dir, exist_ok=True)
    #         path = os.path.join(
    #             model_dir, f"model_{cfg.id.count}_ep{i}_{total_steps}.h5"
    #         )
    #         nn.save_weights(path, overwrite=True)
    #         results_dir = os.path.join(cfg.train.results_dir, f"{cfg.id.count}")
    #         os.makedirs(results_dir, exist_ok=True)
    #         filename = f"{cfg.id.count}_{total_steps}"
    #         log_results(results_dir, filename, data_collect, loss_log)

    #     if loss_log != []:
    #         logger.info(f"Loss after episode: {loss_log[-1]}")
    #     logger.info(f"Steps before done: {ep_steps}")

    # print(f"Total steps: {total_steps}, Loss: {loss_log[-1]}")
    # env.close()


if __name__ == "__main__":
    main()
