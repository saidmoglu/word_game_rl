import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
from Trie import Trie
import string

def random_letter_short(count):
     # Given letter frequencies
    letter_frequencies = {
        'E': 12.02, 'T': 9.10, 'A': 8.12, 'I': 7.31, 'N': 6.95, 'S': 6.28, 'R': 6.02, 'H': 5.92, 'D': 4.32, 'L': 3.98, 
    }

    # Normalize the frequencies to sum to 1
    total = sum(letter_frequencies.values())
    normalized_frequencies = {letter: freq / total for letter, freq in letter_frequencies.items()}

    # Select letters randomly, with probability proportional to their frequency
    letters, probabilities = zip(*normalized_frequencies.items())
    return np.random.choice(letters, count, p=probabilities)

# return random letters wrt their frequency in English
def random_letters(count):
    # Given letter frequencies
    letter_frequencies = {
        'E': 12.02, 'T': 9.10, 'A': 8.12, 'O': 7.68, 'I': 7.31,
        'N': 6.95, 'S': 6.28, 'R': 6.02, 'H': 5.92, 'D': 4.32,
        'L': 3.98, 'U': 2.88, 'C': 2.71, 'M': 2.61, 'F': 2.30,
        'Y': 2.11, 'W': 2.09, 'G': 2.03, 'P': 1.82, 'B': 1.49,
        'V': 1.11, 'K': 0.69, 'X': 0.17, 'Q': 0.11, 'J': 0.10, 'Z': 0.07
    }

    # Normalize the frequencies to sum to 1
    total = sum(letter_frequencies.values())
    normalized_frequencies = {letter: freq / total for letter, freq in letter_frequencies.items()}

    # Select letters randomly, with probability proportional to their frequency
    letters, probabilities = zip(*normalized_frequencies.items())
    return np.random.choice(letters, count, p=probabilities)

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(self, render_mode=None, size=5):
        self.size = size
        
        self.values = {
        'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4, 'I': 1,
        'J': 8, 'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3, 'Q': 10, 'R': 1,
        'S': 1, 'T': 1, 'U': 1, 'V': 4, 'W': 4, 'X': 8, 'Y': 4, 'Z': 10
        }

        self.words_definitions = self.read_words()
        self.words = self.words_definitions.keys()
        # letter_frequencies = {
        #     'E': 12.02, 'T': 9.10, 'A': 8.12, 'I': 7.31, 'N': 6.95,
        # }
        # lls = list(letter_frequencies.keys())
        # count = 0
        # for word in self.words:
        #     if(all([(x in lls) for x in word])):
        #         count += 1
        # print(f"all words count with these letters: {count}")
        self.max_word_len = len(max(self.words, key=len))
        self.trie = self.build_trie()

        # game state: agent location
        # board
        # current word
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "board": spaces.Text(max_length=size*size,min_length=size*size,charset=string.ascii_uppercase),
                "word" : spaces.Text(max_length=self.max_word_len,min_length=0,charset=string.ascii_uppercase),
                "active": spaces.MultiBinary(1), # whether the word is being selected or not
            }
        )

        # Actions: 8 directions, start selecting, stop selecting
        self.action_space = spaces.Discrete(10)

        """
        The following dictionary maps abstract actions from `self.action_space` to
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
            4: np.array([1, 1]),
            5: np.array([1, -1]),
            6: np.array([-1, 1]),
            7: np.array([-1, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window_size = 512
        self.window = None
        self.clock = None
    
    def read_words(self):
        with open('words.txt') as f:
            lines = f.readlines()
            words_definitions = {key: value for key, value in (item.split("\t", 1) for item in lines[2:])}
            return words_definitions

    def build_trie(self):
        trie = Trie()
        for word in self.words:
            trie.insert(word)
        return trie
    
    def _get_obs(self):
        return {
            "agent": self._agent_location, 
            "board": self._board,
            "word" : self._word,
            "active": self._active,
        }
    
    def _get_info(self):
        return {
            '_inactive_penalty_count':self._inactive_penalty_count,
            '_invalid_move_count':self._invalid_move_count,
            '_invalid_word_count':self._invalid_word_count,
            '_too_long_word_count':self._too_long_word_count,
            '_already_active_penalty_count':self._already_active_penalty_count,
            '_already_inactive_penalty_count':self._already_inactive_penalty_count,
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        self._board = ''.join(random_letter_short(self.size**2))
        self._agent_location = np.array([0,0])
        self._word = ''
        self._active = 0
        self._reward = 0

        self._inactive_count = 0 # how long have we been in inactive state
        self._inactive_penalty_count = 0
        self._invalid_move_count = 0
        self._invalid_word_count = 0
        self._too_long_word_count = 0
        self._already_active_penalty_count = 0
        self._already_inactive_penalty_count = 0

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info
    
    def step(self, action):
        self._reward = 0
        terminated = False
        if(action < 8):
            direction = self._action_to_direction[action]
            new_agent_location = self._agent_location + direction
            # if going out of bounds, negative reward, no change in game state
            if(np.any(new_agent_location<0) | np.any(new_agent_location>=self.size)):
                self._reward = -10
                self._invalid_move_count += 1
            else:
                # for valid move, move the agent
                self._agent_location = new_agent_location
                # add new letter if active
                if(self._active):
                    # don't add if the word is max possible, negative reward in this case
                    if(len(self._word) == self.max_word_len):
                        self._reward = -10
                        self._too_long_word_count += 1
                    else:
                        row = self._agent_location[0]
                        col = self._agent_location[1]
                        self._word = self._word + self._board[row*self.size+col]
                else:
                    self._inactive_count += 1
                    if(self._inactive_count >= self.size):
                        self._reward = -10
                        self._inactive_penalty_count += 1
        if(action == 8):
            # activate. If already active, negative reward.
            if(self._active):
                self._reward = -10
                self._already_active_penalty_count += 1
            else:
                self._active = True
                row = self._agent_location[0]
                col = self._agent_location[1]
                self._word = self._board[row*self.size+col]
        if(action == 9):
            # if already inactive, just negative reward.
            if(not self._active):
                self._reward = -10
                self._already_inactive_penalty_count += 1
            # if agent was active, check if the word is valid. Otherwise negative reward.
            else:
                self._active = False
                self._inactive_count = 0
                self._reward = -10
                if self.trie.search(self._word) and len(self._word)>2:
                    self._reward = len(self._word) * sum(self.values[l] for l in self._word)
                    terminated = True
                else:
                    self._invalid_word_count += 1
                self._word = ""
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        return observation, self._reward, terminated, False, info

    def render(self):
        if self.render_mode == "ansi":
            return self._render_frame()
    
    def _render_frame(self):
        if self.render_mode == "human":
            if self.window is None:
                pygame.init()
                pygame.display.init()
                self.my_font = pygame.font.SysFont('arial', 30)
                self.window = pygame.display.set_mode(
                    (self.window_size, self.window_size)
                )
            if self.clock is None:
                self.clock = pygame.time.Clock()

            canvas = pygame.Surface((self.window_size, self.window_size))
            canvas.fill((255, 255, 255))
            pix_square_size = (
                self.window_size / self.size / 2
            )  # The size of a single grid square in pixels

            # Draw the board
            for i in range(self.size):
                for j in range(self.size):
                    text_image = self.my_font.render(self._board[i*self.size+j],False,(0,0,0))
                    canvas.blit(text_image,((i+0.1)*pix_square_size,(j+0.1)*pix_square_size))
                    # pygame.draw.rect(canvas, (255, 0, 0),
                    #     pygame.Rect(pix_square_size * self._target_location,(pix_square_size, pix_square_size),),)
            # Now we draw the agent
            agent_color = (0, 0, 255) if self._active else (200,200,255)
            pygame.draw.circle(
                canvas,
                agent_color,
                (self._agent_location + 0.5) * pix_square_size,
                pix_square_size / 6,
            )

            # write the current word
            text_image = self.my_font.render(self._word,False,(0,0,0))
            canvas.blit(text_image,(self.size*pix_square_size,self.size*pix_square_size))

            # write the reward for the move
            text_image = self.my_font.render(f"reward: {self._reward}",False, (0,0,0))
            canvas.blit(text_image,(self.size*pix_square_size, (self.size+1)*pix_square_size))

            # Finally, add some gridlines
            for x in range(self.size + 1):
                pygame.draw.line(
                    canvas,
                    0,
                    (0, pix_square_size * x),
                    (self.window_size, pix_square_size * x),
                    width=3,
                )
                pygame.draw.line(
                    canvas,
                    0,
                    (pix_square_size * x, 0),
                    (pix_square_size * x, self.window_size),
                    width=3,
                )
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # ansi
            result = "-" * self.size * 3 + "\n"
            for i in range(self.size):
                for j in range(self.size):
                    if(j==0):
                        result += '|'
                    result += self._board[i*self.size+j]
                    if(np.all(self._agent_location == np.array([i,j]))):
                        result += '+' if self._active else 'o'
                    else:
                        result += ' '
                    result += '|'
                    if(j==self.size-1):
                        result += '\n' + "-" * self.size * 3 + "\n"
            result += "word: " + self._word + "\n"
            result += "reward: " + self._reward + "\n"
            return result

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()