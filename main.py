import numpy as np
import cv2
import tensorflow as tf
# # 1.23.1'
# file_content = np.load("LeagueAIDQN/save-00000756/replay-buffer/frames.npy",allow_pickle=True)
# print()
# # Print the above file content
# print("The above file content:")
# print(file_content.shape)
#
import tensorflow as tf
DQN = tf.keras.models.load_model('DRQNmodelSave/LeagueAIDRQN/save-00003088/dqn.h5')
DQN.summary()
# target_dqn = tf.keras.models.load_model('LeagueAI4/save-00002692/target_dqn.h5')
# target_dqn.summary()

def process_frame(frame, shape=(84, 84)):
    """Preprocesses a 210x160x3 frame to 84x84x1 grayscale
    Arguments:
        frame: The frame to process.  Must have values ranging from 0-255
    Returns:
        The processed frame
    """
    frame = frame.astype(np.uint8)  # cv2 requires np.uint8, other dtypes will not work

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = frame[34:34 + 160, :160]  # crop image
    frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
    frame = frame.reshape((*shape, 1))

    return frame
n=0
while n<20:
    img0 = cv2.imread(r'D:\FinalProject\Image\testImage\Screen02.png')
    state = [np.expand_dims(process_frame(img0), axis=0) for _ in range(4)]
    state = np.concatenate(state, axis=0)
    q_vals = DQN.predict(state.reshape((-1, 4, 84, 84, 1)))[0]
    print(q_vals)
    n+=1
# for x in range(0,1000):
#     y = np.random.randint(0,13)
#     print(y)
# from collections import defaultdict
#
# player_list = []
# state = {
#     'stats': {
#         'kills': 0,
#         'deaths': 0,
#         'assists': 0,
#         'minion_kills': 0,
#         'health': 0,
#         'mana': 0,
#         'opponent_health': 100,
#         'Q': 0,
#         'W': 0,
#         'E': 0,
#         'R': 0,
#         'D': True,
#         'F': True
#     },
#
#     'positions': defaultdict(lambda: None)
# }
#
# # playerState = state['stats']
# #
# # player_list.append(dict(zip([1], [playerState])))
# # player_list.append(dict(zip([2], [playerState])))
# # print(player_list)
# #
# # np.save('D:\FinalProject\data2\playerState.npy', player_list, allow_pickle=True, fix_imports=True)
