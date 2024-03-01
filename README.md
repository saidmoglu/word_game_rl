# Reinforcement learning for word game

My ultimate goal is to train an agent that can play Spell Tower game on IOS, in Zen mode, scoring very high points. I first start with this simpler game: We have an n by n grid of letters. Starting from a letter of your choice, you can go to any of the 8 neighboring letters and keep going this way to form a word. For simplicity we allow reusing letters. The score, if the word is a valid English word, is the sum of letter points times the length of the word.

![image](https://github.com/saidmoglu/word_game_rl/assets/17039179/27cd66a4-33c0-4eb0-b809-1f6e6646d7b4)

In this grid we can make the word FATHER starting from F. Letter points are as follows:
```
'A': 1, 'B': 3, 'C': 3, 'D': 2, 'E': 1, 'F': 4, 'G': 2, 'H': 4, 'I': 1,
'J': 8, 'K': 5, 'L': 1, 'M': 3, 'N': 1, 'O': 1, 'P': 3, 'Q': 10, 'R': 1,
'S': 1, 'T': 1, 'U': 1, 'V': 4, 'W': 4, 'X': 8, 'Y': 4, 'Z': 10
```
Thus, FATHER would earn 12*6=72 points.

I implement the game as an environment in gymnasium (OpenAI gym). My initial implementation of the environment has the following details:
#### State (observation):
```
"agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
"board": spaces.Text(max_length=size*size,min_length=size*size,charset=string.ascii_uppercase),
"word" : spaces.Text(max_length=self.max_word_len,min_length=0,charset=string.ascii_uppercase),
"active": spaces.MultiBinary(1), # whether the word is being selected or not
```
Agent is the location of the agent on the board. Board is the entire board and word is the current selection. Active is whether the player is making a selection or not, this is needed for the player to move to the starting letter of the desired word.

#### Action space
Size is 10. 8 for the 8 directions, 1 for activate and 1 for deactivate. While playing the mobile game, the human player will put their finger down on a starting letter (activate) and after moving around, will lift their finger (deactivate). As soon as finger is lifted, current selection is evaluated: If it is a word in the dictionary, then it is scored as explained above.

#### Rewards
- Going out of bounds is -10. 
- If active and word is at max length, don’t add any new letter, and reward is -10.
- If active and agent tries to activate again (same as inactive and deactivate actions) -10.
- Deactivate, if word is invalid, -10. If word is valid, reward is the word score.

DQN implementation: For the RL part, I made use of DQN implementation based on examples online.

#### Initial result 
I can train the agent to get a score of 0. In this case, the agent simply moves back and forth between 2 adjacent letters, without ever activating. So the agent didn’t learn to make any actual words and scoring points.

### Follow up
Then I tried following:
1- Negative reward for moving around in inactive state after a while. We want the agent to attempt to make words.
2- Improve board generation: Instead of random letters, pick common letters more often to increase the number of valid words on the board. Most such games do this, a fully random board (i.e. uniformly picked letters) has very few possible words.
3- Add more info to print out such as invalid move count, inactive move count, valid word count etc. to see what is happening in the learning process.

Result:
Looking at the printed values, it seems valid word count is increasing very slowly: Out of 10K episodes, only 621 ended due to a valid word. The rest has ended due to the step limit, which is 50. At this rate the model will have a very hard time learning actual words.
Essentially, it is very unlikely for actual words to be made by random exploration, thus the model is unable to learn actual words.
Average score remains negative. Penalties mostly come from the invalid words. The good this is that the model learns other things pretty well: Doesn’t do invalid actions, doesn’t spend too much time in inactive state, and doesn’t try to create too long words.

### Follow up
I tried using only a subset of letters, so the model doesn’t need to learn a lot of new words.

Result:
I restricted the alphabet to 5 letters, ETAIN. Watching the gameplay, the model learned a few basic words such as tit, tat, tet, nan etc all made using a back and forth between the vowel and consonant letter. They are worth 9 points each. The model also learned to make a max length word to get the least possible penalty.
Setting the alphabet back to the full size, and we get the old behavior back, in fact, valid word count goes down to 0 during training, the model just learns to make the longest possible words to minimize its penalty.
