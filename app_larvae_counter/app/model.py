import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .preProcess import PreProcess





class Model():
    def __init__(self):
        self.loaded_model = self.loadModel()

    def loadModel(self):
        model_filename = "src/counter/counter/new_svm.pkl"
        loadModel = joblib.load(model_filename)
        return loadModel

    def classifyImage(self, siftList):
        # reducedSiftList = self.performPCA(siftList)
        # print((siftList))

        resultList = []
        for sift in siftList:
            # 

            
            # sift_2d = sift.reshape(1, -1)

            X_feature = sift.astype(float).reshape(1, -1)

            X_feature = PreProcess.normalize_data(X_feature)

            # print(X_feature)

            y = self.loaded_model.predict(X_feature)
        
            prediction = 1 if y=="larvae" else 0
            resultList.append(prediction)

        count = sum(1 for element in resultList if element == 1)
        numElements = len(siftList)
        consensus = count/numElements *100
        print(f"Percentage of positive: {consensus:.2f}%")

        if consensus <= 51: # Voting process could be improved
            return 0
        else:
            return 1
    @staticmethod
    def performPCA(sift):
                
        # Extract the descriptors and convert them into a matrix
        descriptors_matrix = np.array(sift)

        # Step 2: Normalize the data
        scaler = StandardScaler()
        descriptors_matrix_normalized = scaler.fit_transform(descriptors_matrix)

        # Step 3: Compute the covariance matrix
        covariance_matrix = np.cov(descriptors_matrix_normalized, rowvar=False)

        # Step 4: Perform PCA
        n_components = 32  # Choose the number of principal components you want to retain
        pca = PCA(n_components=n_components)
        pca.fit(descriptors_matrix_normalized)

        # Step 5: Transform the data
        reduced_data = pca.transform(descriptors_matrix_normalized)

        return reduced_data






