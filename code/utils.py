import numpy as np
from sklearn.model_selection import learning_curve

def get_learning_curve(model,name):
  train_size, train_score, test_score = learning_curve(estimator=model, X=X, y=y, cv=10 )
  train_score_m = np.mean(train_score, axis=1)
  test_score_m = np.mean(test_score, axis=1)
  plt.plot(train_size, train_score_m, 'o-', color="b")
  plt.plot(train_size, test_score_m, 'o-', color="r")
  plt.legend(('Training score', 'Test score'), loc='best')
  plt.xlabel("Training Samples")
  plt.ylabel("Score")
  title_text = "Learning curve for "+name
  plt.title(title_text)
  plt.grid()
  plt.show()


# Scoring function

def get_scores(arr):
  TP = arr[0][0]
  FP = arr[1][1]
  TN = arr[1][0]
  FN = arr[0][1]
  acc = (TP+FP)/(TP+FP+TN+FN)
  pre = TP/(TP+FP)
  rec = TP/(TP+FN)
  f1 = 2*((pre*rec)/(pre+rec))
  return acc, pre, rec, f1

# Print Accuracy

def get_results(cmatrix,scores):
  print(cmatrix)
  print("Mean Accuracy is                     :",np.mean(scores))
  print("Standard Deviation of accuracies is  :",np.std(scores))
  cmatrix = cmatrix.to_numpy()
  return cmatrix
