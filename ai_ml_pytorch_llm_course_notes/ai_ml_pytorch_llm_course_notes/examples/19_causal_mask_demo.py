"""因果掩码：当前位置只能看到自己和之前的 token。"""
import numpy as np

seq_len = 6
allow = np.tril(np.ones((seq_len, seq_len), dtype=int))
additive_mask = np.where(allow == 1, 0.0, -np.inf)
print('0/1 allow mask:\n', allow)
print('\nadditive mask used before softmax:\n', additive_mask)
