
# Python program to convert a list to string
def listToString(s):
    # initialize an empty string
    str1 = ""
    # traverse in the string
    for ele in s:
        str1 += ele + ' '
    # return string
    return str1

# Python program to convert a list to string
def listToString_blank(s):
    # initialize an empty string
    str1 = ""
    # traverse in the string
    for ele in s:
        str1 += ele + ''
    # return string
    return str1

# Python program to convert a list to string
def stringToList(s):
    # initialize an empty string
    str1 = []
    # traverse in the string
    for ele in range(int(len(s)/2)):
        str1.append(s[ele*2])
    # return string
    return str1

def mean(arr):
    return sum(arr) / len(arr)

def cross_correlation(x, y):
    # Calculate means
    x_mean = mean(x)
    y_mean = mean(y)

    # Calculate numerator
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))

    # Calculate denominators
    x_sq_diff = sum((a - x_mean) ** 2 for a in x)
    y_sq_diff = sum((b - y_mean) ** 2 for b in y)
    denominator = math.sqrt(x_sq_diff * y_sq_diff)
    correlation = numerator / denominator

    return correlation


def find_keys_by_value(dictionary, value):
    return [key for key, val in dictionary.items() if val == value]


def find_values_by_key(dictionary, keys):
    return [dictionary[key] for key in keys]


# def model_preprocessing_function(examples):
#     return model_tokenizer(examples['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH)


def fnirs_process(current_dir, input_folder):

    # column_name = ['AB_I_O', 'AB_PHI_O', 'AB_I_DO', 'AB_PHI_DO', 'CD_I_O', 'CD_PHI_O', 'CD_I_DO', 'CD_PHI_DO', 'label']
    # data_df = pd.DataFrame([], columns=column_name)

    data_all_transformed, target_all_transformed = [], []

    subject_seg = 426
    seg_length = 210 # sampling rate is 5.2Hz, one trial is 2 second length,
    for file_i in range(len(input_folder)): #2):#
        directory = input_folder[file_i]  ## 57: Haptics  41: FordA  33: Earthquakes  24: DistalPhalanxTW
        input_data = current_dir + "/" + directory

        ##   data reading
        data_subject = pd.read_csv(input_data, header=None).to_numpy()
        data_subject = data_subject[1:, :]
        data_subject_size = data_subject.shape

        subject_seg_temp = 0
        seg_length_temp = 0
        AA = np.zeros([subject_seg - 6, data_subject_size[1] - 1])
        BB = np.zeros([seg_length, data_subject_size[1] - 1])
        data_subject_transformed = np.zeros(
            [int(data_subject_size[0] / subject_seg) * int(subject_seg / seg_length), data_subject_size[1] - 1, seg_length])
        target_subject_transformed = np.zeros([int(data_subject_size[0] / subject_seg) * int(subject_seg / seg_length)],
                                              dtype=int)
        for i_subject_seg in range(int(data_subject_size[0] / subject_seg)):
            AA = data_subject[i_subject_seg * subject_seg + 3:(i_subject_seg + 1) * subject_seg - 3, :]
            subject_seg_temp += 1
            for i_seg_length in range(int(subject_seg / seg_length)):
                BB = AA[i_seg_length * seg_length:(i_seg_length + 1) * seg_length, :-1]
                data_subject_transformed[seg_length_temp, :, :] = np.transpose(BB)
                target_subject_transformed[seg_length_temp] = AA[i_seg_length * seg_length, -1]
                seg_length_temp += 1

        data_all_transformed.extend(data_subject_transformed)
        target_all_transformed.extend(target_subject_transformed)

    return np.array(data_all_transformed), np.array(target_all_transformed)


def fnirs_process_slide_window(current_dir, input_folder):

    # column_name = ['AB_I_O', 'AB_PHI_O', 'AB_I_DO', 'AB_PHI_DO', 'CD_I_O', 'CD_PHI_O', 'CD_I_DO', 'CD_PHI_DO', 'label']
    # data_df = pd.DataFrame([], columns=column_name)

    data_all_transformed, target_all_transformed = [], []

    subject_seg = 150  # sampling rate is 5.2Hz, we set 30 second length,
    for file_i in range(len(input_folder)): #2):#
        directory = input_folder[file_i]
        input_data = current_dir + "/" + directory

        ##   data reading
        data_subject = pd.read_csv(input_data, header=None).to_numpy()
        data_subject = data_subject[1:, :]

        data_subject_size = data_subject.shape

        subject_seg_temp = 0
        AA = np.zeros([subject_seg, data_subject_size[1]])
        data_subject_transformed = np.zeros([int(data_subject_size[0] / subject_seg), subject_seg, data_subject_size[1] - 2])
        target_subject_transformed = np.zeros([int(data_subject_size[0] / subject_seg)],
                                              dtype=int)
        for i_subject_seg in range(int(data_subject_size[0] / subject_seg)):
            AA = data_subject[i_subject_seg * subject_seg:(i_subject_seg + 1) * subject_seg, :]
            data_subject_transformed[subject_seg_temp, :, :] = AA[:, :-2]   #  np.transpose(BB)
            target_subject_transformed[subject_seg_temp] = AA[75, -1]
            subject_seg_temp += 1

        baseline_index = np.where(target_subject_transformed == 0)
        wm_load3_index = np.where(target_subject_transformed == 3)

        baseline_data_subject_transformed = np.squeeze(data_subject_transformed[baseline_index, :, :])
        wm_load3_data_subject_transformed = np.squeeze(data_subject_transformed[wm_load3_index, :, :])

        baseline_target_subject_transformed = target_subject_transformed[baseline_index]
        wm_load3_target_subject_transformed = np.ones_like(target_subject_transformed[wm_load3_index])

        BinaryClass_data_subject_transformed = np.concatenate((baseline_data_subject_transformed, wm_load3_data_subject_transformed), axis = 0)
        BinaryClass_dtarget_subject_transformed = np.concatenate((baseline_target_subject_transformed, wm_load3_target_subject_transformed), axis = 0)

        data_all_transformed.extend(BinaryClass_data_subject_transformed)
        target_all_transformed.extend(BinaryClass_dtarget_subject_transformed)

    return np.array(data_all_transformed), np.array(target_all_transformed)
