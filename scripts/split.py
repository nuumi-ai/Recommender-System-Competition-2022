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

# only keep users that have more than 10 reviews
review_df = review_df[review_df.groupby('user_profile')['user_profile'].transform('size') >= 10]

# only keep places that have more than 20 reviews
review_df = review_df[review_df.groupby('place_id')['place_id'].transform('size') >= 20]

user_list = review_df['user_profile'].unique()
place_list = review_df['place_id'].unique()

# user_map: user_profile -> user_index
user_map = {}
for i in range(len(user_list)):
    user_map[user_list[i]] = i

# place_map: place_id -> place_index
place_map = {}
for i in range(len(place_list)):
    place_map[place_list[i]] = i

# add colums user_index and place_index to review_df
review_df['user_index'] = review_df['user_profile'].map(lambda i: user_map[i])
review_df['place_index'] = review_df['place_id'].map(lambda i: place_map[i])

# add colums place_index to restaurant_df
restaurant_df = restaurant_df[restaurant_df['place_id'].isin(place_list)]
restaurant_df['place_index'] = restaurant_df['place_id'].map(lambda i: place_map[i])

# reset index
review_df = review_df[['user_index', 'place_index', 'rating', 'publish_time', 'review_text']].reset_index(drop=True)

# dataset: 0 -> train set, 1 -> public test set, 2 -> private test set
review_df['dataset'] = 0

test_public_user_set = set()
test_private_user_set = set()
for i, row in review_df.iterrows():
    if row['user_index'] not in test_public_user_set:
        test_public_user_set.add(row['user_index'])
        review_df.loc[i, 'dataset'] = 1
        continue
    if row['user_index'] not in test_private_user_set:
        test_private_user_set.add(row['user_index'])
        review_df.loc[i, 'dataset'] = 2
        continue

train = review_df[review_df['dataset'] == 0]
test_public = review_df[review_df['dataset'] == 1]
test_private = review_df[review_df['dataset'] == 2]

# remove place that is not in the train set
place_list = train['place_index'].unique()
test_public = test_public[test_public['place_index'].isin(place_list)]
test_private = test_private[test_private['place_index'].isin(place_list)]
restaurant_df = restaurant_df[restaurant_df['place_index'].isin(place_list)]

print('the length of train is ' + str(len(train)))
print('the length of test_public is ' + str(len(test_public)))
print('the length of test_private is ' + str(len(test_private)))

test = pd.concat([test_public, test_private])

# save to tsv
train_columns_name = ['user_index', 'place_index', 'rating', 'publish_time', 'review_text']
train = train[train_columns_name].reset_index(drop=True)
train.to_csv('./competition-data/train.tsv', sep='\t')

test_columns_name = ['user_index', 'place_index', 'rating', 'publish_time', 'review_text']
test = test[test_columns_name].reset_index(drop=True)
test.to_csv('./competition-data/test_with_rating.tsv', sep='\t')

test_columns_name = ['user_index', 'place_index', 'publish_time', 'review_text']
test = test[test_columns_name].reset_index(drop=True)
test.to_csv('./competition-data/test.tsv', sep='\t')

restaurant_columns_name = ['place_index', 'name', 'category', 'related_categories', 'description',
                           'priceRange', 'address', 'rating', 'reviews', 'menu', 'open_hours',
                           'domain', 'popular_times', 'is_food_and_beverage', 'reviews_tags']
restaurant_df = restaurant_df[restaurant_columns_name].reset_index(drop=True)
restaurant_df.to_csv('./competition-data/restaurant.tsv', sep='\t')
