from copy import deepcopy

def apply_to_window(df, window_idx, column_key,
                    function, function_kwargs={},
                    window_name='window_id'):
    results = {}
    for w in range(window_idx):
        key = f'{w}'
        temp_data = df[(df[window_name] == w)][
            column_key]
        if temp_data.empty:
            results[key] = None
        else:
            temp_results = function(temp_data, **function_kwargs)
            results[key] = temp_results
    return results

def map_back_to_window(data, result_dic, new_col_name, column_key='window_id'):
    datacopy = deepcopy(data)
    datacopy[new_col_name] = [result_dic.get(str(int(val))) for val in datacopy[column_key]]
    return datacopy


def map_cols_to_window(result_dict, prefix, n_colums, existing_df):
    copy_data = deepcopy(existing_df)
    colnames = [f'{prefix}_{val}' for val in range(n_colums)]
    new_df = pd.DataFrame(result_dict.values(), index=result_dict.keys(),
                          columns=colnames).reset_index(drop=True)
    return pd.concat([copy_data, new_df], axis=1)


if __name__ == '__main__':
    # this is simply to show how this function is supposed to work

    import pandas as pd
    import os
    import numpy as np
    base_dir = os.getcwd()
    data_dir = 'data/asg_two'
    file = 'HAR_train.csv'
    file_to_path = os.path.join(base_dir, data_dir, file)
    data = pd.read_csv(os.path.join(file_to_path))
    data['window_id'] = np.random.randint(0, 50, len(data))

    def mean_f(data, shift):
        return sum(data) / len(data) + shift

    apply_to_window(data, 8, 50, 'acc_z', mean_f, function_kwargs={'shift': 5})
