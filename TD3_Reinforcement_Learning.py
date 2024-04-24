import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
import gym
import cogflow

#PluginMnaager manages plugins such as MlflowPlugin, KubeflowPlugin, and DatasetPlugin.
cml = cogflow.PluginManager()

#To print all available attributes and methods of a PluginManager module
#print(dir(cml))

#To get the documentation string (docstring) for the PluginManager module
#help(cml)

#Default all plugins will be in deactivated status
cml.plugin_status()

#Activate only required plugin to make use of it, this activation is effective for user session only
mlp_activate = cml.activate_plugin("MlflowPlugin")
kfp_activate = cml.activate_plugin("KubeflowPlugin")
dsp_activate = cml.activate_plugin("DatasetPlugin")

#To get plugin instances
mlp = cml.get_mlflow_plugin()
kfp = cml.get_kflow_plugin()
dsp = cml.get_dataset_plugin()


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        print("State shape ", self.state.shape)
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(mlp.pyfunc.PythonModel):
    def __init__(
            self,
            state_dim,
            action_dim,
            max_action,
            discount=0.99,
            tau=0.005,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                    torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                    self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + not_done * self.discount * target_q

        # Get current Q estimates
        current_q1, current_q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    '''
    This is for support inference engine for our QCode
    '''

    def predict(self, context, model_input):
        return self.select_action(self, model_input)

    def save(self, filename):
        cirtic_filename = filename + "_critic"
        cirtic_optimizer_filename = filename + "_critic_optimizer"
        actor_filename = filename + "_actor"
        actor_optimizer_filename = filename + "_actor_optimizer"
        mlp.pytorch.save_state_dict(self.critic.state_dict(), cirtic_filename)
        mlp.pytorch.save_state_dict(self.critic_optimizer.state_dict(), cirtic_optimizer_filename)

        mlp.pytorch.save_state_dict(self.actor.state_dict(), actor_filename)
        mlp.pytorch.save_state_dict(self.actor_optimizer.state_dict(), actor_optimizer_filename)
        return {
            "cirticFileName": cirtic_filename,
            "ciritcOptimizerFileName": cirtic_optimizer_filename,
            "actorFileName": actor_filename,
            "actorOptimizerFileName": actor_optimizer_filename
        }

    def load(self, filename):
        cirtic_filename = filename + "_critic"
        cirtic_optimizer_filename = filename + "_critic_optimizer"
        actor_filename = filename + "_actor"
        actor_optimizer_filename = filename + "_actor_optimizer"
        self.critic.load_state_dict(torch.load(cirtic_filename))
        self.critic_optimizer.load_state_dict(torch.load(cirtic_optimizer_filename))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(actor_filename))
        self.actor_optimizer.load_state_dict(torch.load(actor_optimizer_filename))
        self.actor_target = copy.deepcopy(self.actor)

    def load_context(self, context):
        model_file_path = context.artifacts["model"]
        load(model_file_path)


### Evaluator
# Runs policy for X episodes and returns average reward

# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.reset(seed=seed + 100)

    avg_reward = 0.
    for _ in range(eval_episodes):
        state, done = eval_env.reset()[0], False
        while not done:
            action = policy.select_action(np.array(state))
            nstep = eval_env.step(action)
            state, reward, terminated, truncated, info = nstep
            done = truncated or terminated
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


def trainModel(policy="TD3",
               envType="HalfCheetah-v2",
               seed: int = 0,
               start_timesteps: int = 25e3,
               eval_freq: int = 5e3,
               max_timesteps: int = 1e6,
               expl_noise: float = 0.1,
               batch_size: int = 256,
               discount: float = 99,
               tau: float = 0.005,
               policy_noise: float = 0.2,
               noise_clip: float = 0.5,
               policy_freq: int = 2,
               load_model=""
               ):
    # import TD3
    file_name = f"{policy}_{envType}_{seed}"
    print("---------------------------------------")
    print(f"Policy: {policy}, Env: {envType}, Seed: {seed}")
    print("---------------------------------------")

    experiment_id = mlp.set_experiment(
        experiment_name="Custom Models TD3",

    )
    mlp.pytorch.autolog()
    mlp.autolog()

    env = gym.make(envType)

    # Set seeds
    env.reset(seed=seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print("state_dim", env.observation_space.shape, "action_dim", env.action_space.shape,
          "actionShape", env.action_space.sample().shape)
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": discount,
        "tau": tau,
    }

    # Initialize policy
    if policy == "TD3":
        # Target policy smoothing is scaled wrt the action scale
        kwargs["policy_noise"] = policy_noise * max_action
        kwargs["noise_clip"] = noise_clip * max_action
        kwargs["policy_freq"] = policy_freq
        policy = TD3(**kwargs)

    if load_model != "":
        policy_file = file_name if load_model == "default" else load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = ReplayBuffer(state_dim, action_dim)
    print("starting evaluation untrained ploicy")
    # Evaluate untrained policy
    evaluations = [eval_policy(policy, envType, seed)]

    (state, _), done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    print("cogflow is starting")

    with mlp.start_run(run_name="custom_model_run") as run:
        mlp.log_param("EnviromentType", envType)
        mlp.log_param("Seed", seed)
        mlp.log_param("Start timesteps", start_timesteps)
        mlp.log_param("Evaluation frequency", eval_freq)
        mlp.log_param("Max timesteps", max_timesteps)
        mlp.log_param("Expl noise", expl_noise)
        mlp.log_param("Batch size", batch_size)
        mlp.log_param("Discount", discount)
        mlp.log_param("Tau", tau)
        mlp.log_param("Policy noise", policy_noise)
        mlp.log_param("Noise clip", noise_clip)
        mlp.log_param("Policy Frequency", policy_freq)
        for t in range(int(max_timesteps)):
            episode_timesteps += 1
            # Select action randomly or according to policy
            if t < start_timesteps:
                action = env.action_space.sample()
            else:
                action = (
                        policy.select_action(np.array(state))
                        + np.random.normal(0, max_action * expl_noise, size=action_dim)
                ).clip(-max_action, max_action)

            # Perform action
            next_state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

            # Store data in replay buffer
            replay_buffer.add(state, action, next_state, reward, done_bool)

            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if t >= start_timesteps:
                policy.train(replay_buffer, batch_size)

            if done:
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(
                    f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
                mlp.log_metric("Reward", episode_reward, step=episode_num)
                # Reset environment
                (state, _), done = env.reset(), False
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

                # Evaluate episode
            if (t + 1) % eval_freq == 0:
                episod_eval_reward = eval_policy(policy, envType, seed)
                evaluations.append(episod_eval_reward)
                mlp.log_metric("Episode Evaluation Reward", episod_eval_reward, step=int((t + 1) // eval_freq))
                np.save(f"./results/{file_name}", evaluations)
                articats = policy.save(f"./models/{file_name}")

                model_info = mlp.pyfunc.log_model(
                    artifact_path=file_name,
                    python_model=policy,
                    artifacts=articats,
                    pip_requirements=[],
                    input_example=state,
                    signature=mlp.models.infer_signature(next_state, action)
                )


trainModel()
