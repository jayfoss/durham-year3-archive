import numpy as np
import pandas as pd
import re
import os
import surprise as sp
import ast
import datetime
import math

DATASET_ROOT = './'
BNB_DATASET = './bnb_filtered_data/'
REC_DATA = './rec_data/'

JSON_BUSINESS_DATA = 'yelp_academic_dataset_business.json'
JSON_CHECKIN_DATA = 'yelp_academic_dataset_checkin.json'
JSON_REVIEW_DATA = 'yelp_academic_dataset_review.json'
JSON_TIP_DATA = 'yelp_academic_dataset_tip.json'
JSON_USER_DATA = 'yelp_academic_dataset_user.json'
JSON_COVID_FEATURES_DATA = 'yelp_academic_dataset_covid_features.json'

BUSINESS_DEPENDENT_DATASETS = [JSON_CHECKIN_DATA, JSON_REVIEW_DATA, JSON_TIP_DATA, JSON_COVID_FEATURES_DATA]

def create_data_directory():
    if not os.path.exists(BNB_DATASET):
        os.mkdir(BNB_DATASET)
        return True
    else:
        return False

def create_rec_directory():
    if not os.path.exists(REC_DATA):
        os.mkdir(REC_DATA)
        return True
    else:
        return False

def load_data_raw(path, chunksize=None):
    return pd.read_json(r'' + DATASET_ROOT + path, lines=True, chunksize=chunksize)

def load_data_filtered(path, chunksize=None):
    return pd.read_json(r'' + BNB_DATASET + path, lines=True, chunksize=chunksize)


def view_business_categories():
    b = load_business_raw()
    l = b.categories.dropna().str.split(',').explode().str.strip().unique()
    l.sort()
    print(l.tolist())

def filter_business_to_bnb(b):
    f = ['Bed & Breakfast', 'Hotels,', 'Hotel bar']
    f = [re.escape(g) for g in f]
    f.append('Hotels$')
    f = '|'.join(f)
    b = b[b.categories.str.contains(f, na=False)]
    return b

def write_bnb_dataframe(b, path):
    b.to_json(r'' + BNB_DATASET + path, orient='records', lines=True)

def write_bnb_dataframe_handle(b, handle):
    b.to_json(handle, orient='records', lines=True)

#Filter all Yelp data to only include businesses we are interested in. Save those to file
def initial_setup():
    no_bnb_directory = create_data_directory()
    if not no_bnb_directory:
        return False
    print('Filtered data directory not found. Running initial setup... This may take a long time')
    b = load_data_raw(JSON_BUSINESS_DATA)
    b = filter_business_to_bnb(b)
    write_bnb_dataframe(b, JSON_BUSINESS_DATA)
    business_ids = b['business_id']
    b = None
    s = pd.Series([],dtype=pd.StringDtype())
    for dset in BUSINESS_DEPENDENT_DATASETS:
        with open(r'' + BNB_DATASET + dset, 'a') as f:
            with load_data_raw(dset, 1000) as reader:
                for chunk in reader:
                    filtered = chunk[chunk['business_id'].isin(business_ids)]
                    if dset != JSON_CHECKIN_DATA and dset != JSON_COVID_FEATURES_DATA:
                        s = s.append(filtered['user_id'])
                    write_bnb_dataframe_handle(filtered, f)
    users_interacted = s.unique()
    business_ids = None
    s = None
    with open(r'' + BNB_DATASET + JSON_USER_DATA, 'a') as f:
        with load_data_raw(JSON_USER_DATA, 1000) as reader:
            for chunk in reader:
                filtered = chunk[chunk['user_id'].isin(users_interacted)]
                write_bnb_dataframe_handle(filtered, f)

def read_reviews():
    return load_data_filtered(JSON_REVIEW_DATA)

def read_users():
    return load_data_filtered(JSON_USER_DATA)

#EXPERIMENT. Create a list of ratings
def generate_rating_list(review_data):
    users = {}
    for v in review_data.user_id.values:
        users[v] = len(users) - 1
    businesses = {}
    for v in review_data.business_id.values:
        businesses[v] = len(businesses) - 1

    stars = []
    for index, row in review_data.iterrows():
        stars.append((users[row['user_id']], businesses[row['business_id']], row['stars']))
    return (users, businesses, stars)

#EXPERIMENT. Calculate mean rating for all items/users
def calculate_mean(ubs_tuple):
    users, businesses, stars = ubs_tuple
    sum_ratings = 0
    for u, b, rating in stars:
        sum_ratings += rating
    rating_mean = sum_ratings / len(stars)
    return rating_mean

#EXPERIMENT. A naive svd implementation
def train_svd_u(ubs_tuple, latent_factors = 50, epochs = 100, learn_rate = 0.0007):
    users, businesses, stars = ubs_tuple
    p = np.random.normal(0, .1, (len(users), latent_factors))
    q = np.random.normal(0, .1, (len(businesses), latent_factors))

    for epoch in range(epochs):
        for user_index, business_index, rating in stars:
            err = rating - np.dot(p[user_index], q[business_index])
            el = err * learn_rate

            p[user_index] += el * q[business_index]
            q[business_index] += el * p[user_index]
    return p, q

#EXPERIMENT. A better regularized SVD implementation with biases
def train_svd(ubs_tuple, mean, latent_factors = 100, epochs = 20, learn_rate = 0.005, reg = 0.02):
    users, businesses, stars = ubs_tuple
    p = np.random.normal(0, .1, (len(users), latent_factors))
    q = np.random.normal(0, .1, (len(businesses), latent_factors))
    b_u = np.zeros(len(users))
    b_b = np.zeros(len(businesses))

    for epoch in range(epochs):
        for user_index, business_index, rating in stars:
            err = rating - (mean + b_u[user_index] + b_b[business_index] + np.dot(p[user_index], q[business_index]))
            
            p_u = p[user_index]
            b_u[user_index] = b_u[user_index] + learn_rate * (err - reg * b_u[user_index])
            b_b[business_index] = b_b[business_index] + learn_rate * (err - reg * b_b[business_index])
            p[user_index] = p[user_index] + learn_rate * (err * q[business_index] - reg * p[user_index])
            q[business_index] = q[business_index] + learn_rate * (err * p_u - reg * q[business_index])
            
    return p, q

def rmse(x, y):
    return np.sqrt(np.mean((y - x) ** 2))

#EXPERIMENT. Evaluate our SVD implementation
def evaluate_svd(ubs_tuple, p, q):
    stars = ubs_tuple[2]
    input_list = []
    output_list = []
    for u, b, rating in stars:
        input_list.append(rating)
        output_list.append(np.dot(p[u], q[b]))
    return rmse(np.array(input_list), np.array(output_list))

#Grid search for an SVD. Used only when creating the filter initially to optimize hyperparameters
def surprise_svd_grid(reviews):
    reader = sp.Reader(rating_scale=(1, 5))
    data = sp.Dataset.load_from_df(reviews[['user_id', 'business_id', 'stars']], reader)
    svd = sp.SVD(50, 20)
    grid_params = {
        'n_epochs': [50],
        'n_factors': [35, 40],
        'lr_all': [0.005, 0.006],
        'reg_all': [0.04, 0.05]
    }
    A = sp.model_selection.GridSearchCV(sp.SVD, grid_params, ['rmse', 'mae'], cv=5, refit=True, n_jobs=-2, joblib_verbose=10)
    A.fit(data)
    return A

#Create an SVD model-based collaborative filter using surprise
def surprise_svd(reviews):
    reader = sp.Reader(rating_scale=(1, 5))
    data = sp.Dataset.load_from_df(reviews[['user_id', 'business_id', 'stars']], reader)
    svd = sp.SVD(40, 50, lr_all = 0.005, reg_all = 0.05)
    cross = sp.model_selection.cross_validate(svd, data, ['rmse'], cv = 5)
    return (svd, cross, data)

def restore_collab_model():
    print('Restoring model')
    _, svd = sp.dump.load(REC_DATA + 'collab_model')
    return svd

def save_collab_model(svd_model):
    sp.dump.dump(REC_DATA + 'collab_model', algo=svd_model)

#Setup for the collaborative model. Either restore from file or train a new one if not found
def build_collab_model(reviews):
    create_rec_directory()
    model = None
    if os.path.exists(REC_DATA + 'collab_model'):
        model = restore_collab_model()
    if model is None:
        print('No saved model found. Training...')
        A, c, svd_data = surprise_svd(reviews)
        save_collab_model(A)
        model = A
    return model

#Create the matrix of businesses and attributes for collaborative filtering
def generate_ba_matrix(businesses):
    attributes = {}
    business_attributes = []
    count = -1
    for attr in businesses['attributes'].tolist():
        count += 1
        business_attributes.append({})
        if attr is None:
            continue
        for attr_name in attr:
            if attr[attr_name] == 'True' or attr[attr_name] == 'False':
                attributes[attr_name] = None
                business_attributes[count][attr_name] = True if attr[attr_name] == 'True' else False
            elif attr[attr_name].isnumeric():
                attributes[attr_name + '_' + attr[attr_name]] = None
                business_attributes[count][attr_name + '_' + attr[attr_name]] = True
            else:
                if attr[attr_name][0] == '{' and '}' in attr[attr_name]:
                    try:
                        t = ast.literal_eval(attr[attr_name])
                        for sub in t:
                            if sub.lower() == 'none' or sub.lower() == 'no':
                                continue
                            attributes[attr_name + '_' + sub] = None
                            business_attributes[count][attr_name + '_' + sub] = t[sub]
                    except:
                        pass
                else:
                    sub = attr[attr_name]
                    if attr[attr_name][0] == 'u' and attr[attr_name][1] == '\'':
                        sub = sub[1:]
                    sub = re.sub(r'\W+', '', sub)
                    if sub.lower() != 'none' and sub.lower() != 'no':
                        attributes[attr_name + '_' + sub] = None
                        business_attributes[count][attr_name + '_' + sub] = True
                        
    ba_matrix = np.zeros((len(business_attributes), len(attributes)))
    for i, b in enumerate(business_attributes):
        for j, a in enumerate(attributes):
            if a in b:
                ba_matrix[i][j] = 1
    return (ba_matrix, attributes)

#Get ratings for the user. Unrated items will remain zero
def get_business_ratings_for_user(user_id, businesses, reviews):
    b_ratings = np.zeros(len(businesses))
    reviews_by_user = reviews.loc[reviews['user_id'] == user_id]
    for i in reviews_by_user.index:
        ind = businesses.loc[businesses['business_id'] == reviews_by_user['business_id'][i]]
        b_ratings[ind.index] = reviews_by_user['stars'][i]
    return b_ratings

#Create ratings using content based filtering
def content_based_filter(b_ratings, ba_matrix, attributes, businesses):
    user_profile = np.zeros(len(attributes))
    for i, _ in enumerate(user_profile):
        user_profile[i] = np.dot(ba_matrix.T[i], b_ratings)

    row_sums = ba_matrix.sum(axis=1)
    np.divide(ba_matrix, row_sums[:, np.newaxis], out=ba_matrix, where=row_sums[:, np.newaxis] != 0)

    pred_ratings_cbf = np.zeros(len(businesses))
    for i, row in enumerate(ba_matrix):
        pred_ratings_cbf[i] = np.dot(row, user_profile)

    maxi = pred_ratings_cbf.max()
    if maxi != 0:
        pred_ratings_cbf = np.interp(pred_ratings_cbf, (pred_ratings_cbf.min(), maxi), (1, 5))
    return pred_ratings_cbf

def get_n_top_indices(ratings, n, randomize=False):
    if randomize:
        n = n * 2
    largest = np.argpartition(ratings, -n)[-n:]
    if randomize:
        np.random.shuffle(largest)
    return largest[:10]

def get_n_top_ratings(ratings, n, randomize=False):
    return ratings[get_n_top_indices(ratings, n, randomize)]

#Run the collaborative filter for all businesses
def collaborative_filter(A, user_id, businesses):
    pred_ratings_clf = np.zeros(len(businesses))
    for i, b_id in enumerate(businesses['business_id'].tolist()):
        pred_ratings_clf[i] = A.predict(user_id, b_id).est
    return pred_ratings_clf

#Combine RS0 and RS1 using the weighting hybrid method. This isn't a good combination
def hybrid_filter_weighted(cbf, clf, a, b):
    wcbf = cbf * a
    wclf = clf * b
    combined = wcbf + wclf
    maxi = combined.max()
    if maxi != 0:
        combined = np.interp(combined, (combined.min(), maxi), (1, 5))
    return combined

#Combine RS0 and RS1 using the switching hybrid method
def hybrid_filter_switching(cbf, clf, b_ratings):
    non_zero_b_ratings = np.nonzero(b_ratings)
    f_b_ratings = b_ratings[non_zero_b_ratings]
    f_pred_ratings_cbf = cbf[non_zero_b_ratings]
    f_pred_ratings_clf = clf[non_zero_b_ratings]
    rmse_clf = rmse(f_b_ratings, f_pred_ratings_clf)
    rmse_cbf = rmse(f_b_ratings, f_pred_ratings_cbf)
    #Calculate a 95% confidence interval for the rmse
    ci_setting = 1.96
    clf_ci = ci_setting * np.sqrt(np.abs((rmse_clf * (1 - rmse_clf)) / len(f_pred_ratings_clf)))
    cbf_ci = ci_setting * np.sqrt(np.abs((rmse_cbf * (1 - rmse_cbf)) / len(f_pred_ratings_cbf)))
    #Choose the recommender with a smaller range resulting from the confidence interval
    if cbf_ci < clf_ci:
        return cbf
    return clf

#Determine rmse and ci for RS0, RS1 and hybrid. We remove the unrated items at this stage so a complete b_ratings array can be passed
def grade_combined_filter(b_ratings, pred_ratings_cbf, pred_ratings_clf, combined):
    non_zero_b_ratings = np.nonzero(b_ratings)
    f_b_ratings = b_ratings[non_zero_b_ratings]
    f_pred_ratings_cbf = pred_ratings_cbf[non_zero_b_ratings]
    f_pred_ratings_clf = pred_ratings_clf[non_zero_b_ratings]
    f_combined = combined[non_zero_b_ratings]
    rmse_clf = rmse(f_b_ratings, f_pred_ratings_clf)
    rmse_cbf = rmse(f_b_ratings, f_pred_ratings_cbf)
    rmse_hybrid = rmse(f_b_ratings, f_combined)
    ci_setting = 1.96
    clf_ci = ci_setting * np.sqrt(np.abs((rmse_clf * (1 - rmse_clf)) / len(f_pred_ratings_clf)))
    cbf_ci = ci_setting * np.sqrt(np.abs((rmse_cbf * (1 - rmse_cbf)) / len(f_pred_ratings_cbf)))
    hybrid_ci = ci_setting * np.sqrt(np.abs((rmse_hybrid * (1 - rmse_hybrid)) / len(f_combined)))
    return {
        'collaborative': rmse_clf,
        'content_based': rmse_cbf,
        'hybrid': rmse_hybrid,
        'hybrid_improvement_collaborative': rmse_clf - rmse_hybrid,
        'hybrid_improvement_content_based': rmse_cbf - rmse_hybrid,
        'collaborative_improvement_content_based': rmse_cbf - rmse_clf,
        'collaborative_ci': clf_ci,
        'content_based_ci': cbf_ci,
        'hybrid_ci': hybrid_ci,
        'hybrid_is_improved': (rmse_clf - rmse_hybrid) > 0
    }

"""
Calculate the precision and recall for the predictions. This is implemented based on the reference implementation
of precision and recall in the Surprise library FAQ. https://surprise.readthedocs.io/en/stable/FAQ.html.
The Surprise version of the function has been modified to make it more generic so that it can be reused for content-based/hybrid analysis.
Numpy has been used to get the average precision and recall since that is all we are interested in.
"""
def calc_pr(predictions, n, threshold = 3.5):
    user_predictions = {}
    found_businesses = {}
    for prediction in predictions:
        if prediction[0] not in user_predictions:
            user_predictions[prediction[0]] = []
        user_predictions[prediction[0]].append((prediction[3], prediction[2], prediction[1]))
        if prediction[1] not in found_businesses:
            found_businesses[prediction[1]] = len(found_businesses)
    precision_list = []
    recall_list = []

    for user, ratings in user_predictions.items():
        ratings.sort(key = lambda x: x[0], reverse = True)
        relevant = sum((rui >= threshold) for (_, rui, _) in ratings)
        recommended = sum((rhat >= threshold) for (rhat, _, _) in ratings[:n])
        rec_rel = sum(((rui >= threshold) and (rhat >= threshold)) for (rhat, rui, _) in ratings[:n])

        add = 0
        if recommended > 0:
            add = rec_rel / recommended
        precision_list.append(add)

        add = 1
        if relevant > 0:
            add = rec_rel / relevant
        recall_list.append(add)
    return (np.array(precision_list).mean(), np.array(recall_list).mean(), user_predictions, found_businesses)

#Grade the RS for collaborative, content-based and a hybrid. Hybrid is either weighted or switched
def grade_all_combined_filter(A, businesses, reviews, hybrid_weighted = False):
    np.random.seed(42)
    reviews_test = reviews.sample(1000)
    A = surprise_svd(reviews_test)[0]
    review_test_business_ids = reviews_test['business_id'].unique().tolist()
    businesses_test = businesses.loc[businesses['business_id'].isin(review_test_business_ids)]
    businesses_test = businesses_test.reset_index()
    ba_matrix, attributes = generate_ba_matrix(businesses_test)
    clf_grades = []
    cbf_grades = []
    hybrid_grades = []
    clf_ci = []
    cbf_ci = []
    hybrid_ci = []
    collaborative_test = A.test(A.trainset.build_testset())
    content_based_test = []
    combined_test = []
    clf_novelty = []
    cbf_novelty = []
    hybrid_novelty = []
    clf_precision, clf_recall, user_keys, business_keys = calc_pr(collaborative_test, 10)
    bus_test_usr_prop = []
    for b in businesses_test['business_id'].tolist():
        no_raters = len(reviews.loc[reviews['business_id'] == b]['user_id'].unique().tolist())
        prop = -math.log2(no_raters / len(user_keys))
        bus_test_usr_prop.append(prop)
    for user_id, values in user_keys.items():
        b_ratings = get_business_ratings_for_user(user_id, businesses_test, reviews_test)
        non_zero_b_ratings = np.nonzero(b_ratings)
        if len(non_zero_b_ratings[0]) == 0:
            continue            
        pred_ratings_cbf = content_based_filter(b_ratings, ba_matrix, attributes, businesses_test)
        pred_ratings_clf = collaborative_filter(A, user_id, businesses_test)
        combined = None
        if hybrid_weighted:
            combined = hybrid_filter_weighted(pred_ratings_cbf, pred_ratings_clf, 0.25, 0.75)
        else:
            combined = hybrid_filter_switching(pred_ratings_cbf, pred_ratings_clf, b_ratings)
        for _, _, business in values:
            b_ind = businesses_test.index[businesses_test['business_id'] == business][0]
            content_based_test.append((user_id, business, b_ratings[b_ind], pred_ratings_cbf[b_ind]))
            combined_test.append((user_id, business, b_ratings[b_ind], combined[b_ind]))
        grade = grade_combined_filter(b_ratings, pred_ratings_cbf, pred_ratings_clf, combined)
        clf_grades.append(grade['collaborative'])
        cbf_grades.append(grade['content_based'])
        hybrid_grades.append(grade['hybrid'])
        clf_ci.append(grade['collaborative_ci'])
        cbf_ci.append(grade['content_based_ci'])
        hybrid_ci.append(grade['hybrid_ci'])
        top_cbf = get_n_top_indices(pred_ratings_cbf, 10)
        top_clf = get_n_top_indices(pred_ratings_clf, 10)
        top_combined = get_n_top_indices(combined, 10)
        for j in range(0, 3):
            var = top_cbf
            if j == 1:
                var = top_clf
            elif j == 2:
                var = top_combined
            nv = 0
            for i in var:
                nv += bus_test_usr_prop[i]
            novelty = nv / 10
            if j == 0:
                cbf_novelty.append(novelty)
            elif j == 1:
                clf_novelty.append(novelty)
            elif j == 2:
                hybrid_novelty.append(novelty)
    cbf_precision, cbf_recall, _, _ = calc_pr(content_based_test, 10)
    hybrid_precision, hybrid_recall, _, _ = calc_pr(combined_test, 10)
    clf_grades = np.array(clf_grades).mean()
    cbf_grades = np.array(cbf_grades).mean()
    hybrid_grades = np.array(hybrid_grades).mean()
    clf_ci = np.array(clf_ci).mean()
    cbf_ci = np.array(cbf_ci).mean()
    hybrid_ci = np.array(hybrid_ci).mean()
    clf_novelty = np.array(clf_novelty).mean()
    cbf_novelty = np.array(cbf_novelty).mean()
    hybrid_novelty = np.array(hybrid_novelty).mean()
    return {
        'collaborative': clf_grades,
        'content_based': cbf_grades,
        'hybrid': hybrid_grades,
        'hybrid_improvement_collaborative': clf_grades - hybrid_grades,
        'hybrid_improvement_content_based': cbf_grades - hybrid_grades,
        'collaborative_improvement_content_based': cbf_grades - clf_grades,
        'collaborative_ci': clf_ci,
        'content_based_ci': cbf_ci,
        'hybrid_ci': hybrid_ci,
        'hybrid_is_improved': (clf_grades - hybrid_grades) > 0,
        'hybrid_weighted': hybrid_weighted,
        'clf_precision': clf_precision,
        'cbf_precision': cbf_precision,
        'hybrid_precision': hybrid_precision,
        'clf_recall': clf_recall,
        'cbf_recall': cbf_recall,
        'hybrid_recall': hybrid_recall,
        'clf_f1': (2 * clf_recall * clf_precision) / (clf_recall + clf_precision),
        'cbf_f1': (2 * cbf_recall * cbf_precision) / (cbf_recall + cbf_precision),
        'hybrid_f1': (2 * hybrid_recall * hybrid_precision) / (hybrid_recall + hybrid_precision),
        'clf_novelty': clf_novelty,
        'cbf_novelty': cbf_novelty,
        'hybrid_novelty': hybrid_novelty
    }

#Remove any businesses that are closed due to covid
def filter_covid_closures(predictions, businesses):
    covid_features = load_data_filtered(JSON_COVID_FEATURES_DATA)
    for i, b in enumerate(businesses['business_id'].tolist()):
        f = covid_features.loc[covid_features['business_id'] == b]
        if len(f.index) > 0 and f['Temporary Closed Until'].iloc[0] != 'FALSE':
            try:
                closure = datetime.datetime.strptime(f['Temporary Closed Until'].iloc[0], '%Y-%m-%dT%H:%M:%S.%fZ')
                now = datetime.datetime.utcnow()
                if closure > now:
                    predictions[i] = -1
            except ValueError:
                pass
    return predictions

#Get the best available predictions for the given state
def get_best_for_state(n, predictions, businesses, state):
    pred = np.copy(predictions)
    in_state = businesses.loc[businesses['state'] == state]
    out_state = businesses.loc[businesses['state'] != state]
    best = []
    while len(best) < n:
        i = pred.argmax()
        if pred[i] == -1:
            break
        if i in in_state.index:
            best.append(i)
        pred[i] = -1
    pred = np.copy(predictions)
    best_out = []
    #If enough predictions couldn't be found for the state, fill remaining with non-state options
    while len(best) + len(best_out) < n:
        i = pred.argmax()
        if pred[i] == -1:
            break
        if i in out_state.index:
            best_out.append(i)
        pred[i] = -1
    return best, best_out

def get_recommended_business_names(recommended, businesses):
    found = businesses.iloc[recommended]
    found = found['name'] + ', ' + found['state']
    return found.tolist()

#Primary function for producing a recommendation for given user with desired state
def recommend_to_user(user_id, A, businesses, reviews, state):
    ba_matrix, attributes = generate_ba_matrix(businesses)
    b_ratings = get_business_ratings_for_user(user_id, businesses, reviews)

    pred_ratings_cbf = content_based_filter(b_ratings, ba_matrix, attributes, businesses)
    pred_ratings_clf = collaborative_filter(A, user_id, businesses)

    combined = hybrid_filter_switching(pred_ratings_cbf, pred_ratings_clf, b_ratings)
    combined = filter_covid_closures(combined, businesses)
    best_combined = []
    best_out = []
    if state == 'any':
        best_combined = list(get_n_top_indices(combined, 10))
    else:
        best_combined, best_out = get_best_for_state(10, combined, businesses, state)
    return get_recommended_business_names(best_combined + best_out, businesses), len(best_combined)

def pretty_print_list(l):
    for e in l:
        print(e)

#Get states as a comma separated list
def get_state_list_string(states):
    res = ''
    for i, s in enumerate(states):
        if i != len(states) - 1:
            res += s + ', '
        else:
            res += s
    return res

#States available for post-filtering context-awareness
def get_states(businesses):
    return businesses['state'].unique().tolist()

ui_running = True
initial_setup()
businesses = load_data_filtered(JSON_BUSINESS_DATA)
reviews = read_reviews()
states = get_states(businesses)
A = build_collab_model(reviews)
available_users = reviews['user_id'].unique().tolist()

while ui_running:
    print('There are ' + str(len(available_users)) + ' users in the system.')
    user_index = input('Enter a number between 0 and ' + str(len(available_users) - 1) + ' inclusive to \'login\' as that user and see their recommendations, or type \'exit\' to quit.\n')
    if user_index == 'exit':
        break
    elif user_index == 'grade':
        print(grade_all_combined_filter(A, businesses, reviews, False))
        continue
    elif user_index == 'gradew':
        print(grade_all_combined_filter(A, businesses, reviews, True))
        continue
    try:
        user_index = int(user_index)
    except ValueError:
        print('Invalid user index entered')
        continue
    if user_index < 0 or user_index > len(available_users) - 1:
        print('Invalid user index entered')
        continue
    print('Welcome! Please note, no further information will be collected from you except what you explicitly enter here')
    print('We have hotels/hotel bars in the following states: ' + get_state_list_string(states))
    state = input('Please choose a state by entering its 2-character code from the above choices or type \'any\' to view establishments from all states\n')
    if state not in states and state != 'any':
        print('Sorry, there are currently no hotels/hotel bars in this state listed with us')
        continue
    user_id = available_users[user_index]
    recommended, num_recommendations = recommend_to_user(user_id, A, businesses, reviews, state)
    print('================')
    print('Recommendations found for your state search:\n')
    pretty_print_list(recommended[:num_recommendations])
    if num_recommendations < len(recommended):
        print('\nUnfortunately, we couldn\'t find more hotels/hotel bars matching your state.')
        print('Here are some others you may like:\n')
        pretty_print_list(recommended[num_recommendations:])
    print('================')
    print('How were this ratings generated? We looked at all places you have reviewed in the past and then tried to find similar items. We also looked at what other people, that liked similar items to you, had positive opinions of. We then estimated which of these predictions is more likely to be accurate and chose the more accurate one to show you.')
    print('================')
