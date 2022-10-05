import numpy as np
import pandas as pd

# read data
review_columns = ['user_profile', 'place_id', 'rating', 'publish_time', 'review_text']
review_df = pd.read_csv("./data/reviews.tsv", sep="\t")[review_columns]

restaurant_columns = ['place_id', 'name', 'category', 'related_categories', 'description', 'priceRange', 'address',
                      'rating',
                      'reviews', 'menu', 'open_hours', 'domain', 'popular_times', 'is_food_and_beverage',
                      'reviews_tags']
restaurant_df = pd.read_csv("./data/restaurants.tsv", sep="\t")[restaurant_columns]

# shuffle review_df
review_df = review_df.sample(frac=1)

# only keep places that have more than 20 reviews
review_df = review_df[review_df.groupby('place_id')['place_id'].transform('size') >= 20]

# only keep users that have more than 10 reviews
complement_review_df = review_df[review_df.groupby('user_profile')['user_profile'].transform('size') < 10]
review_df = review_df[review_df.groupby('user_profile')['user_profile'].transform('size') >= 10]

# separate review_df into training data, leaderboard test set and private test set
# dataset: 0 -> train set, 1 -> public test set, 2 -> private test set
review_df['dataset'] = 0

test_public_user_set = set()
test_private_user_set = set()
for i, row in review_df.iterrows():
    if row['user_profile'] not in test_public_user_set:
        test_public_user_set.add(row['user_profile'])
        review_df.loc[i, 'dataset'] = 1
        continue
    if row['user_profile'] not in test_private_user_set:
        test_private_user_set.add(row['user_profile'])
        review_df.loc[i, 'dataset'] = 2
        continue

train = review_df[review_df['dataset'] == 0]
test_public = review_df[review_df['dataset'] == 1]
test_private = review_df[review_df['dataset'] == 2]

# remove place that is not in the train set
place_list = train['place_id'].unique()
test_public = test_public[test_public['place_id'].isin(place_list)]
test_private = test_private[test_private['place_id'].isin(place_list)]
restaurant_df = restaurant_df[restaurant_df['place_id'].isin(place_list)]
complement_review_df = complement_review_df[complement_review_df['place_id'].isin(place_list)]

print('the length of train is ' + str(len(train)))
print('the length of test_public is ' + str(len(test_public)))
print('the length of test_private is ' + str(len(test_private)))
print('the length of complement data is ' + str(len(complement_review_df)))

user_list = train['user_profile'].unique()
place_list = train['place_id'].unique()

# Complement data contains restaurants that only exist in main trianing data, but more user reviews
complement_review_df = complement_review_df[complement_review_df['place_id'].isin(place_list)]
complement_user_list = complement_review_df['user_profile'].unique()

# user_map: user_profile -> user_index
user_map = {}
for i in range(len(user_list)):
    user_map[user_list[i]] = i

complement_user_map = {}
for i in range(len(complement_user_list)):
    complement_user_map[complement_user_list[i]] = i + len(user_list)

# place_map: place_id -> place_index
place_map = {}
for i in range(len(place_list)):
    place_map[place_list[i]] = i

# add colums user_index and place_index to review_df
train['user_index'] = train['user_profile'].map(lambda i: user_map[i])
train['place_index'] = train['place_id'].map(lambda i: place_map[i])

test_public['user_index'] = test_public['user_profile'].map(lambda i: user_map[i])
test_public['place_index'] = test_public['place_id'].map(lambda i: place_map[i])

test_private['user_index'] = test_private['user_profile'].map(lambda i: user_map[i])
test_private['place_index'] = test_private['place_id'].map(lambda i: place_map[i])

# add colums user_index and place_index to complement_review_df
complement_review_df['user_index'] = complement_review_df['user_profile'].map(lambda i: complement_user_map[i])
complement_review_df['place_index'] = complement_review_df['place_id'].map(lambda i: place_map[i])

# add colums place_index to restaurant_df
restaurant_df = restaurant_df[restaurant_df['place_id'].isin(place_list)]
restaurant_df['place_index'] = restaurant_df['place_id'].map(lambda i: place_map[i])

# save to tsv
train_columns_name = ['user_index', 'place_index', 'rating', 'publish_time', 'review_text']
train = train[train_columns_name].reset_index(drop=True)
train.to_csv('./competition-data/train.tsv', sep='\t', index=False)

train_columns_name = ['user_index', 'place_index', 'rating', 'publish_time', 'review_text']
complement_review_df = complement_review_df[train_columns_name].reset_index(drop=True)
complement_review_df.to_csv('./competition-data/train_complement.tsv', sep='\t', index=False)

test_columns_name = ['user_index', 'place_index', 'publish_time', 'rating']
test_public = test_public[test_columns_name].reset_index(drop=True)
test_public.to_csv('./competition-data/test_leaderboard.tsv', sep='\t', index=False)
test_public.drop(['rating'], axis=1).to_csv('./competition-data/test_leaderboard_wo_rating.tsv', sep='\t', index=False)

test_private = test_private[test_columns_name].reset_index(drop=True)
test_private.to_csv('./competition-data/test_private.tsv', sep='\t', index=False)

restaurant_columns_name = ['place_index', 'name', 'category', 'related_categories', 'description',
                           'priceRange', 'address', 'rating', 'reviews', 'menu', 'open_hours',
                           'domain', 'popular_times', 'is_food_and_beverage', 'reviews_tags']
restaurant_df = restaurant_df[restaurant_columns_name].reset_index(drop=True)
restaurant_df.to_csv('./competition-data/restaurant.tsv', sep='\t', index=False)
