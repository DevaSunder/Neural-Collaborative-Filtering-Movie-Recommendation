from flask import Flask, render_template, request
import pandas as pd
import torch
import sys
import sys
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import preprocessing
from torch.utils.data import Dataset, TensorDataset, DataLoader

ratings_cols = ['UserID', 'MovieID', 'Rating', 'Timestamp']
ratings = pd.read_csv('./ml-1m/ml-1m/ratings.dat', sep='::', engine='python', names=ratings_cols)

num_users = ratings.UserID.unique().shape[0]
num_items = ratings.MovieID.unique().shape[0]
#print('no. users: %d, no. items: %d' %(num_users, num_items))

class CustomDataset(Dataset):
    def __init__(self, users, items, y):
        self.x = torch.cat([
            torch.LongTensor(users).unsqueeze(0).transpose(0, 1),
            torch.LongTensor(items).unsqueeze(0).transpose(0, 1)
        ], axis=1)
        self.y = torch.FloatTensor(y)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)


le1 = preprocessing.LabelEncoder()
le2 = preprocessing.LabelEncoder()

batch_size = 256

train_dataset = CustomDataset(
    le1.fit_transform(ratings.UserID),
    le2.fit_transform(ratings.MovieID),
    ratings.Rating.values
)
train_loader = DataLoader(train_dataset, batch_size=batch_size)


# Define the NCF model
class NCF(nn.Module):
    def __init__(self, num_users, num_items, emb_size=128, hidden_size=256):
        super(NCF, self).__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.mlp = nn.Sequential(
            nn.Linear(emb_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.user_emb.weight.data.uniform_(0, 0.05)
        self.item_emb.weight.data.uniform_(0, 0.05)

    def forward(self, data):
        x, y = data[:, :1], data[:, 1:]
        u, v = self.user_emb(x), self.item_emb(y)
        uv = torch.cat((u, v), dim=1)
        return self.mlp(uv.view(uv.size(0), -1)).squeeze()

model = NCF(num_users, num_items, emb_size=256, hidden_size=256).to(device)

learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()

model.eval()

ratings['le_UserID'] = le1.transform(ratings.UserID)
ratings['le_MovieID'] = le1.transform(ratings.MovieID)

model.load_state_dict(torch.load('model_weights.pth'))

class TopNCF:
    def __init__(self, model, user_encoder, movie_encoder, movie_dataset):
        self.model = model
        self.user_encoder = user_encoder
        self.movie_encoder = movie_encoder
        self.movie_dataset = movie_dataset

    def recommend_movies(self, user_id, top_k=10):
        user_tensor = torch.LongTensor([self.user_encoder.transform([user_id])[0]]).to(device)
        movie_tensor = torch.LongTensor(range(self.movie_encoder.classes_.shape[0])).to(device)

        user_tensor = user_tensor.repeat(len(movie_tensor), 1)

        data = torch.cat((user_tensor, movie_tensor.unsqueeze(1)), dim=1)
        predictions = self.model(data).detach().cpu().numpy()

        top_indices = predictions.argsort(axis=0)[-top_k:][::-1]
        top_movie_ids = self.movie_encoder.classes_[top_indices]

        return top_movie_ids.tolist()

    def get_top_movie_names(self, user_id, top_k=10):
        top_movie_ids = self.recommend_movies(user_id, top_k)
        movie_names = []
        for movie_id in top_movie_ids:
            movie_title = self.movie_dataset.loc[self.movie_dataset['MovieID'] == movie_id, 'Title'].values[0]
            movie_names.append(movie_title)
        return movie_names

# Define the Flask app
app = Flask(__name__, static_folder='static')



# Define the route for the home page
@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')

# Define the route for the recommendations page
@app.route('/recommend', methods=['GET','POST'])
def recommendations():
    user_id = int(request.form['user_id'])

    # Load the movies dataset from a .dat file into a pandas DataFrame
    movies_cols = ['MovieID', 'Title', 'Genres']
    movies_df = pd.read_csv('./ml-1m/ml-1m/movies.dat', sep='::', engine='python', names=movies_cols, encoding='latin-1')
    movie_dataset = movies_df[['MovieID', 'Title']]
    top_ncf = TopNCF(model, le1, le2, movie_dataset)
    top_movies = top_ncf.get_top_movie_names(user_id, top_k=10)
    return render_template('recommendations.html', user_id=user_id, value=top_movies)

@app.route('/trending',methods=['GET'])
def trending():
    return render_template('trendingnow.html')

@app.route('/topmovies',methods=['GET'])
def topmovies():
    return render_template('topmovies.html')

@app.route('/toptvseries',methods=['GET'])
def toptvseries():
    return render_template('toptvseries.html')

@app.route('/upcoming',methods=['GET'])
def upcoming():
    return render_template('Upcoming.html')

@app.route('/index',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/home1',methods=['GET'])
def home1():
    return render_template('Home page.html')


@app.route('/faq',methods=['GET'])
def faq():
    return render_template('FAQ.html')

@app.route('/nayakan',methods=['GET'])
def nayakan():
    return render_template('Nayakan.html')

@app.route('/searching',methods=['GET'])
def searching():
    return render_template('Searching.html')

@app.route('/allmovies',methods=['GET'])
def allmovies():
    return render_template('allmovies.html')

@app.route('/gd',methods=['GET'])
def gd():
    return render_template('gooddoctor.html')

if __name__ == '__main__':
    app.run(debug=True)

