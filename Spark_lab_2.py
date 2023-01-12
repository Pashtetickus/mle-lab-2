import numpy as np
import os
import sys
import gc
import shutil
from datetime import datetime
from io import BytesIO

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

import pyspark
import pyspark.sql.types as T
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark import SparkContext, SparkConf
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql.types import StringType, ArrayType, IntegerType
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
from pyspark.mllib.linalg import Vectors

# create the session
conf = SparkConf()
conf.set("spark.ui.port", "4050")
conf.set("spark.app.name", "mlops_lab_2_danilov")
conf.set("spark.master", "local")
conf.set("spark.executor.cores", "12")
conf.set("spark.executor.instances", "1")
conf.set("spark.executor.memory", "16g")
conf.set("spark.locality.wait", "0")
conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
conf.set("spark.kryoserializer.buffer.max", "2000")
conf.set("spark.executor.heartbeatInterval", "6000s")
conf.set("spark.network.timeout", "10000000s")
conf.set("spark.shuffle.spill", "true")
conf.set("spark.driver.memory", "16g")
conf.set("spark.driver.maxResultSize", "16g")

# create the context
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

# сконкатенируем все файлы train разбиения, но по памяти ограничим каждый набор 50000 записей, чтобы не было memoryerror (мы на одной машине с 16gb оперативки)
# Подготовим данные для чтения в `sc.binaryFiles`
# Для выдачи рекомендаций обращаемся к исходным данным по айди пользователя, наиболее похожего на нашего. Если у наиболее похожего юзера не хватает рекомендаций, идем к следующему по схожести. Фильтруем дубликаты, чтобы не рекомендовать фильмы, просмотренные таргетным пользователем.


class RecommederSystem:
    def __init__(self, data_dir='ml-20mx16x32/', data_target_dir='train', mode='train'):
        self.data_dir = data_dir
        self.data_target_dir = data_target_dir
        os.makedirs(self.data_target_dir, exist_ok=True)
        data_files = os.listdir(self.data_dir)
        data_files = [os.path.join(self.data_target_dir, fname) for fname in data_files if mode in fname]
        
        for npz_path in data_files:
            shutil.copyfile(os.path.join(self.data_dir, npz_path.split(os.sep)[1]), npz_path)
        npz_rdd = sc.binaryFiles(self.data_target_dir)
        npz_rdd = npz_rdd.map(lambda l: np.load(BytesIO(l[1]))['arr_0'][:500].astype(int).tolist())\
                        .flatMap(lambda x: x).groupByKey() \
                        .map(lambda x: (int(x[0]), list(x[1])))
        
        schema = T.StructType([T.StructField("user_id", T.IntegerType(), True), 
                               T.StructField("movie_id", T.ArrayType(T.IntegerType()), True)])
        self.dataset = npz_rdd.toDF(schema=schema)
        self.dataset.show()
        
    def calc_tf_idf(self):
        hashingTF = HashingTF(inputCol="movie_id", outputCol="tf_features", numFeatures=1000)
        tf = hashingTF.transform(self.dataset)
        tf.cache()
        idf = IDF(inputCol="tf_features", outputCol="tfidf_features").fit(tf)
        self.tfidf = idf.transform(tf)
    
    def sample_random_user(self):
        """Генерирует id случайного пользователя, для которого нужно сделать рекомендации."""
        unique_users_id = [user.user_id for user in self.dataset.select('user_id').distinct().collect()]
        self.user_id = int(np.random.choice(unique_users_id, 1)[0])
        return self.user_id
    
    def calc_similarity_matrix(self, target_user: int):
        if not hasattr(self, 'tfidf'):
            self.calc_tf_idf()
        mat = IndexedRowMatrix(
            self.tfidf.select("user_id", "tfidf_features")\
            .rdd.map(lambda row: IndexedRow(row.user_id, row.tfidf_features.toArray()))).toBlockMatrix()
        sim_matrix = mat.transpose().toIndexedRowMatrix().columnSimilarities()
        sim_matrix = sim_matrix.entries.filter(lambda x: x.i == target_user or x.j == target_user)
        self.sorted_similarity = sim_matrix.sortBy(lambda x: x.value, ascending=False) \
                            .map(lambda x: IndexedRow(x.j if x.i == target_user else x.i,  Vectors.dense(x.value)))
        self.sim_users_ids = [user_info.index for user_info in self.sorted_similarity.collect()]

    def get_movies_by_user(self, sim_user_id: int, movies_not_to_recom: list):
        most_similar_user_movies = self.dataset.filter(self.dataset.user_id == sim_user_id).select("movie_id").rdd
        most_similar_user_movies = set(most_similar_user_movies.collect()[0].movie_id)
        movies_not_to_recom = set(movies_not_to_recom)
        intersect_with_target_user = set.intersection(most_similar_user_movies, movies_not_to_recom)
        new_movies = most_similar_user_movies - intersect_with_target_user
        new_movies = list(new_movies)
        return new_movies
    
    def get_recommendations(self, target_user_id: int, number_of_reccomendations: int = 30):
        if not self.sim_users_ids:
            return 'Check similar users id list - call calc_similarity_matrix for specified user_id'
        recommended_movies = []
        target_user_movies = self.dataset.filter(self.dataset.user_id == target_user_id).select("movie_id").rdd
        target_user_movies = target_user_movies.collect()[0].movie_id
        while len(recommended_movies) < number_of_reccomendations and len(self.sim_users_ids) != 0:
            next_most_similar_user_id = self.sim_users_ids.pop(0)
            new_movies = self.get_movies_by_user(next_most_similar_user_id, recommended_movies)
            recommended_movies.extend(new_movies)
        recommended_movies = recommended_movies[:number_of_reccomendations]
        return recommended_movies


recsystem = RecommederSystem(data_dir='ml-20mx16x32/', data_target_dir='train', mode='train')

user_id = recsystem.sample_random_user()
print('Id случайного пользователя:', user_id)
recsystem.calc_similarity_matrix(user_id)

recommendations = recsystem.get_recommendations(user_id, number_of_reccomendations=30)
print('Рекомендованые фильмы: \n', recommendations)
