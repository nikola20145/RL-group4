import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class MazeEnv(gym.Env):
    def __init__(self):
        super(MazeEnv, self).__init__()
        self.maze = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 'S', 1, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 1, 1, 1, 0, 1],
            [1, 0, 1, 'G', 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
            [1, 1, 1, 0, 1, 0, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 'E', 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ])

        self.start_pos = [1, 1]
        self.goal_pos = [8, 7]
        self.sub_goal_pos = [5, 3]
        self.agent_pos = self.start_pos

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(low=0, high=1, shape=(10, 10), dtype=int)
        self.reset()

    def reset(self):
        self.agent_pos = self.start_pos
        return self._get_obs()

    def step(self, action):
        next_pos = list(self.agent_pos)

        # Up, Down, Left, Right
        if action == 0:
            next_pos[0] -= 1
        elif action == 1:
            next_pos[0] += 1
        elif action == 2:
            next_pos[1] -= 1
        elif action == 3:
            next_pos[1] += 1

        if self._is_valid_move(next_pos):
            self.agent_pos = next_pos

        done = False
        reward = -1  # Default penalty for each move

        if self.agent_pos == self.sub_goal_pos:
            reward = 10  # Reward for reaching sub-goal
        if self.agent_pos == self.goal_pos:
            reward = 100  # Reward for reaching final goal
            done = True

        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        """Returns the current state of the maze with agent's position."""
        obs = np.copy(self.maze)
        obs[tuple(self.agent_pos)] = 'A'  # Mark agent's current position with 'A'
        return obs

    def _is_valid_move(self, pos):
        """Checks if the move is within bounds and not a wall."""
        if 0 <= pos[0] < self.maze.shape[0] and 0 <= pos[1] < self.maze.shape[1]:
            return self.maze[tuple(pos)] != 1
        return False

    def render(self):
        """Renders the maze with the agent, start, sub-goal, and end goal."""
        maze_to_render = np.copy(self.maze)
        
        # Replace markers for rendering (we'll use text annotations for 'S', 'G', 'E')
        maze_to_render[maze_to_render == 'S'] = 0  # Replace 'S' with 0 (empty space)
        maze_to_render[maze_to_render == 'G'] = 0  # Replace 'G' with 0
        maze_to_render[maze_to_render == 'E'] = 0  # Replace 'E' with 0
        maze_to_render = maze_to_render.astype(int)  # Convert to integer array

        # Mark agent's position with a different integer (e.g., 2)
        maze_to_render[tuple(self.agent_pos)] = 2  # Agent's position (marked with 2)

        # Render using imshow
        plt.imshow(maze_to_render, cmap="cool", origin="upper")
        
        # Add text labels for start, sub-goal, and end goal
        plt.text(1, 1, 'S', ha='center', va='center', color='black', fontsize=12)  # Start
        plt.text(5, 3, 'G', ha='center', va='center', color='black', fontsize=12)  # Sub-goal
        plt.text(8, 7, 'E', ha='center', va='center', color='black', fontsize=12)  # End
        
        plt.title("Maze Environment")
        plt.show()


# Example usage of the environment
if __name__ == "__main__":
    env = MazeEnv()
    
    done = False
    env.render()  # Initial rendering of the maze

    while not done:
        action = env.action_space.sample()  # Take a random action
        _, reward, done, _ = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
        env.render()  # Render after each step
