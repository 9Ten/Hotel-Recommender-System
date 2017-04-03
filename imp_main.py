#--- Import Libraries ---#
import pandas as pd
from sklearn import linear_model
from sklearn import preprocessing

#--- Check Trip_type, Companion ---#
def check_input(trip_type, companion):
    # default = leisure
    check_type = ["business", "leisure", None]
    check_com = ["solo", "couple", "friend", "family", None]
    ok = True
    try:
        check_type.remove(trip_type)
    except:
        ok = False
        print("Trip_type is not valid")
    try:
        check_com.remove(companion)
    except:
        ok = False
        print("Companion is not valid")
    return ok

#--- Part Preprocessing ---#
def check_case(target_id, trip_type, companion, file_loc):
    user_data = pd.read_csv(file_loc)
    group_user = user_data.drop('hotel_name', axis=1)
    target_user = group_user.loc[group_user['user_id'] == target_id]
    if len(target_user) == 0:
        print("Target_user is not valid")
    #--- Check Target_user data ---# 
    else:
        other_user = group_user.loc[group_user['user_id'] != target_id]
        #--- Case No Context ---#
        if (trip_type == None) & (companion == None):
            status = [0, "case_none"]
            data_out = get_main(target_id, target_user,
                                other_user, trip_type, companion, status[0])
            del data_out['others']
            return data_out, status[1]
        else:
            num_target = len(target_user.loc[(target_user['trip_type'] == trip_type) & (
                target_user['companion'] == companion)])
            #--- Case Pass Regression ---#
            if num_target >= 6:
                status = [1, "case_pass_regr"]
                data_out = get_main(target_id, target_user,
                                    other_user, trip_type, companion, status[0])
                del data_out['others']
                return data_out, status[1]
            #--- Case Not Pass Regression ---#
            else:
                status = [-1, "case_not_regr"]
                data_out = get_main(target_id, target_user,
                                    other_user, trip_type, companion, status[0])
                data_out, buff_data = re_get_main(
                    data_out, target_id, user_data, trip_type, companion)
                del data_out['others']
                return data_out, status[1]

def get_main(target_id, target, other, trip_type, companion, status):
    #--- Cal Filter By trip_type, companion ---#
    if status == 1:
        target = target.loc[(target['trip_type'] == trip_type)
                            & (target['companion'] == companion)]
    #--- Cal Not Filter By trip_type, companion ---#
    elif (status == -1) | (status == 0):
        pass
    target_weight = get_weight(target)
    target_rank = get_rank_weight(target_weight)
    group_user = {}
    result = {}
    other_user = other['user_id'].drop_duplicates(keep='first')
    other_user = other_user.values.tolist()
    #--- Cal Weight Into Rank All User ---#
    for data in other_user:
        user = other.loc[other['user_id'] == data]
        other_weight = get_weight(user)
        other_rank = get_rank_weight(other_weight)
        result.update({data: [other_weight, other_rank]})
    group_user.update({'target': {target_id: [target_weight, target_rank]}})
    group_user.update({'others': result})
    #--- Make Neighbors ---#
    group_user_rank = cal_corr(target_id, group_user)
    data_out = get_neighbor(group_user_rank)
    return data_out
    # {
    # 'target': {'user_id': [weight, rank_f]},
    # 'others': {'user_id': [weight, rank_f]},
    # 'neighbors': [[user_id], [corr]]
    # }

#--- Cal Regression ---#
def get_weight(df):
    x = df[['price', 'near_station', 'restaurant', 'entertain',
            'shopping_mall', 'convenience_store']].values.tolist()
    y = df['rating'].values.tolist()
    # Create linear regression object
    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(x, y)
    return regr.coef_

#--- Ordered Weight ---#
def get_rank_weight(weight):
    rank = pd.Series(list(weight)).rank(ascending=False)
    rank_f = rank.values.tolist()
    return rank_f

#--- Cal Spearman's Rank ---#
def cal_corr(target_id, group_user_rank):
    target = pd.Series(group_user_rank['target'][target_id][1])
    others = sorted(list(group_user_rank['others'].keys()))
    list_user = []
    list_corr = []
    for user in others:
        other = pd.Series(group_user_rank['others'][user][1])
        corr = target.corr(other, method='spearman')
        list_user.append(user)
        list_corr.append(corr)
    result = [list_user, list_corr]
    group_user_rank.update({'neighbors': result})
    return group_user_rank

#--- Before Recommendation ---#
def get_neighbor(group_user_rank):
    corr = pd.Series(group_user_rank['neighbors'][1]).rank(ascending=True)
    neighbor = corr.values.tolist()
    group_user_rank['neighbors'].append(neighbor)
    return group_user_rank

#--- Solve Case Not Regression ---#
def re_get_main(data_main, target_id, user_data, trip_type, companion):
    target_data = user_data.loc[(user_data['user_id'] == target_id) & (
        user_data['trip_type'] == trip_type) & (user_data['companion'] == companion)]
    other_id = data_main['neighbors'][0]
    other_rank = data_main['neighbors'][2]
    neighbors = pd.DataFrame({'user_id': other_id, 'rank': other_rank})
    neighbors = neighbors.sort_values(by=('rank'), ascending=True)
    list_neighbors = neighbors['user_id'].values.tolist()

    #--- To Pass Regression ---# 
    buff_data = target_data
    rows_t = len(target_data)
    for other in list_neighbors:
        if len(buff_data) >= 6:
            break
        #--- Filter Context Each *Other User ---#
        other_data = user_data.loc[(user_data['user_id'] == other) & (
            user_data['trip_type'] == trip_type) & (user_data['companion'] == companion)]
        rows_o = len(other_data)
        supply = 6 - rows_t
        if (rows_o == 0):
            continue
        else:
            if rows_o < supply:
                sup_data = other_data[:rows_o + 1]
            else:
                sup_data = other_data[:supply + 1]
        #--- Supplement Into Target_user data---#
        buff_data = buff_data.append(sup_data)

    #--- Recal ---#
    new_weight = get_weight(buff_data)
    new_rank = get_rank_weight(new_weight)
    data_main['target'][target_id] = [new_weight, new_rank]
    data_out = cal_corr(target_id, data_main)
    data_out = get_neighbor(data_out)
    return data_out, buff_data

#--- Part Recommendation ---#
def prediction_rating(data_raw, data_frame, **kwargs):
    data_frame = weight_neighbors(data_frame)
    data_rec = pd.DataFrame()
    neighbors = list(data_frame['user_id'])
    # print('Bug ties data: {}'.format(neighbors))

    # Limit Data 25 records
    #--- Rec No Context ---#
    if (kwargs['trip_type'] is None) & (kwargs['companion'] is None):
        for neighbor in neighbors:
            if len(data_rec) > 25:
                break

            result = data_raw.ix[data_raw['user_id'] == neighbor, [
                'user_id', 'hotel_name', 'rating', 'trip_type', 'companion']]
            data_rec = data_rec.append(result)
    #--- Rec With Context ---#
    else:
        for neighbor in neighbors:
            if len(data_rec) > 25:
                break
            try:
                #--- Filter Context Each *Neighbors ---#
                result = data_raw.ix[(data_raw['user_id'] == neighbor) & (data_raw['trip_type'] == kwargs['trip_type']) & (
                    data_raw['companion'] == kwargs['companion']), ['user_id', 'hotel_name', 'rating', 'trip_type', 'companion']]
            except:
                pass
            #--- Supplement Into Target_user data---#
            data_rec = data_rec.append(result)

    #--- Inner Join data_rec With Weight's Neighbors---#
    data_rec = pd.merge(data_rec.drop(
        ['trip_type', 'companion'], axis=1), data_frame, how='inner', on='user_id')
    groups = data_rec.groupby('hotel_name')
    result1 = []
    result2 = []
    for hotel, group in groups:
        #--- Predict New Rating Into Hotel ---#
        predict_rating = group.apply(lambda row: (
            row.loc['weight'] * row.loc['rating']) / group['weight'].sum(), axis=1).sum()
        result1.append(hotel)
        result2 .append(predict_rating)
    hotel_rec = pd.DataFrame(
        data={'hotel_name': result1, 'predict_rating': result2})
    return hotel_rec

#--- Rec Top k hotel Ordered By predict_rating ---#
def recommendation(target_id, data_rec, top_k):
    topk_hotels = data_rec.sort_values(
        by=('predict_rating'), ascending=False).head(top_k)
    return topk_hotels['hotel_name']

#--- Change Rank's Neighbors(Ordinal) Into Weight's Neighbors ---#
def weight_neighbors(data_frame):
    scale = preprocessing.MinMaxScaler(feature_range=(0.1, 1))
    data_frame['weight'] = scale.fit_transform(data_frame[['weight']])
    return data_frame

if __name__ == '__main__':
    #--- Expect each user: 10 records ---#
    file_loc = 'D:/Desktop/Back-end/user_sample.csv'
    
    #--- Get parameters ---#
    target_id = 'user1'
    trip_type = None
    companion = None
    context = {'trip_type': trip_type, 'companion': companion}
    top_k = 10
    ok = check_input(trip_type, companion)

    if ok:
        data_raw = pd.read_csv(file_loc)
        data_out, status = check_case(
            target_id, trip_type, companion, file_loc)
        data_out = {'user_id': data_out['neighbors'][
            0], 'weight': data_out['neighbors'][2]}
        data_out = pd.DataFrame(data=data_out).sort_values(
            by=('weight'), ascending=False)
        #--- Recommendation ---#
        data_rec = prediction_rating(data_raw, data_out, **context)
        list_rec = recommendation(target_id, data_rec, top_k)

        #--- Show Recommendation ---#
        print('Status: {}\n'.format(status))
        print('Recommended top {} hotels base on [{} {}]: \n{}\n'.format(
            top_k, trip_type, companion, list(list_rec)))
        print('Hotels booked of target user: \n{}\n'.format(
            list(data_raw.ix[data_raw['user_id'] == target_id]['hotel_name'])))
        # Dummy Handling list_rec duplicate
