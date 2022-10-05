import pandas as pd

restaurants = pd.read_csv('./competition-data/restaurant.tsv', sep='\t')
assert len(restaurants) == len(restaurants['place_index'].unique())
print(f"number of restaurants: {len(restaurants)}")
print(f"info in restaurant table: {restaurants.columns}")

train_data = pd.read_csv('./competition-data/train.tsv', sep='\t')
print(f"number of training data: {len(train_data)}")
print(f"info in training data: {train_data.columns}")
assert 'user_name' not in train_data.columns and 'user_profile' not in train_data.columns, \
    f"Containing sensitive user name info"

users_in_train_data = train_data['user_index'].unique()
print(f"number of users in train_data: {len(users_in_train_data)}")

restaurants_in_train_data = train_data['place_index'].unique()
print(f"number of restaurants in train_data: {len(restaurants_in_train_data)}")

review_count = train_data.groupby('user_index')['user_index'].transform('size')
min_count, max_count, average_count = review_count.min(), review_count.max(), review_count.mean()
print(f"review per user, min-max-average: {min_count}-{max_count}-{average_count}")

# user index should be continuos
user_count = len(train_data['user_index'].unique())
max_user_index = train_data['user_index'].max()
assert user_count == max_user_index + 1, \
    f"user index is not continuous: {user_count} users, max index {max_user_index}"

# restaurant index should be continuous
place_count = len(train_data['place_index'].unique())
max_place_index = train_data['place_index'].max()

assert place_count == max_place_index + 1, \
    f"place index is not continuous: {place_count} place, max index {max_place_index}"

# users and restaurants in test should have appeared in training data
test_data_leader_board = pd.read_csv('./competition-data/test_leaderboard.tsv', sep='\t')
users_in_test_data = test_data_leader_board['user_index'].unique()
assert set(users_in_test_data).issubset(set(users_in_train_data)), 'users in test data not in training data'

restaurants_in_test_data = test_data_leader_board['place_index'].unique()
assert set(restaurants_in_test_data).issubset(
    set(restaurants_in_train_data)), 'restaurants in test data not in training data'

# users and restaurants in test should have appeared in training data
test_data_p = pd.read_csv('./competition-data/test_private.tsv', sep='\t')
users_in_test_data = test_data_p['user_index'].unique()
assert set(users_in_test_data).issubset(set(users_in_train_data)), 'users in test data not in training data'

restaurants_in_test_data = test_data_p['place_index'].unique()
assert set(restaurants_in_test_data).issubset(
    set(restaurants_in_train_data)), 'restaurants in test data not in training data'


# rating distribution in training data and test data
def rating_distribution(df):
    counts = [len(df[df['rating'] == i]) for i in [1, 2, 3, 4, 5]]
    print('; '.join([str(c) for c in counts]))


print('training data ratings count')
rating_distribution(train_data)

print('leaderboard test data ratings count')
rating_distribution(test_data_leader_board)

print('private test data ratings count')
rating_distribution(test_data_p)
