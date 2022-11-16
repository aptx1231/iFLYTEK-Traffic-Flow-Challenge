import numpy as np

ids = [
82400,
42186,
8906,
60799,
74660,
23462,
61922,
63374,
60998,
71010,
]

weights = [
0.395652395,
0.429904335,
0.448909748,
0.476128923,
0.446560215,
0.422132795,
0.445689617,
0.43443058,
0.428814059,
0.389050316,
]


sum = np.load('../user_data/tmp_data/{}_y_pred.npy'.format(ids[0], ids[0])) * (1.0 / weights[0])
total = (1.0 / weights[0])
strs = str(ids[0])
for i in range(1, len(ids)):
    tmp = np.load('../user_data/tmp_data/{}_y_pred.npy'.format(ids[i], ids[i]))
    sum += tmp * (1.0 / weights[i])
    total += (1.0 / weights[i])
    strs += ('_' + str(ids[i]))

pred = sum / total
name1 = '../prediction_result/result.npy'
print(name1)
np.save(name1, pred)
print(pred.shape)
