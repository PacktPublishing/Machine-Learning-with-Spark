/*
	This code is intended to be run in the Scala shell. 
	Launch the Scala Spark shell by running ./bin/spark-shell from the Spark directory.
	You can enter each line in the shell and see the result immediately.
	The expected output in the Spark console is presented as commented lines following the
	relevant code

	The Scala shell creates a SparkContex variable available to us as 'sc'
*/

// sed 1d train.tsv > train_noheader.tsv
// load raw data
val rawData = sc.textFile("/PATH/train_noheader.tsv")
val records = rawData.map(line => line.split("\t"))
records.first
// Array[String] = Array("http://www.bloomberg.com/news/2010-12-23/ibm-predicts-holographic-calls-air-breathing-batteries-by-2015.html", "4042", ...

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
val data = records.map { r =>
	val trimmed = r.map(_.replaceAll("\"", ""))
	val label = trimmed(r.size - 1).toInt
	val features = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble)
	LabeledPoint(label, Vectors.dense(features))
}
data.cache
val numData = data.count
// numData: Long = 7395
// note that some of our data contains negative feature vaues. For naive Bayes we convert these to zeros
val nbData = records.map { r =>
	val trimmed = r.map(_.replaceAll("\"", ""))
	val label = trimmed(r.size - 1).toInt
	val features = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble).map(d => if (d < 0) 0.0 else d)
	LabeledPoint(label, Vectors.dense(features))
}

// train a Logistic Regression model
import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.tree.configuration.Algo
import org.apache.spark.mllib.tree.impurity.Entropy

val numIterations = 10
val maxTreeDepth = 5
val lrModel = LogisticRegressionWithSGD.train(data, numIterations)
val svmModel = SVMWithSGD.train(data, numIterations)
// note we use nbData here for the NaiveBayes model training
val nbModel = NaiveBayes.train(nbData) 
val dtModel = DecisionTree.train(data, Algo.Classification, Entropy, maxTreeDepth)

// make prediction on a single data point
val dataPoint = data.first
// dataPoint: org.apache.spark.mllib.regression.LabeledPoint = LabeledPoint(0.0, [0.789131,2.055555556,0.676470588, ...
val prediction = lrModel.predict(dataPoint.features)
// prediction: Double = 1.0
val trueLabel = dataPoint.label
// trueLabel: Double = 0.0
val predictions = lrModel.predict(data.map(lp => lp.features))
predictions.take(5)
// res1: Array[Double] = Array(1.0, 1.0, 1.0, 1.0, 1.0)

// compute accuracy for logistic regression
val lrTotalCorrect = data.map { point =>
  if (lrModel.predict(point.features) == point.label) 1 else 0
}.sum
// lrTotalCorrect: Double = 3806.0

// accuracy is the number of correctly classified examples (same as true label)
// divided by the total number of examples
val lrAccuracy = lrTotalCorrect / numData
// lrAccuracy: Double = 0.5146720757268425

// compute accuracy for the other models
val svmTotalCorrect = data.map { point =>
  if (svmModel.predict(point.features) == point.label) 1 else 0
}.sum
val nbTotalCorrect = nbData.map { point =>
  if (nbModel.predict(point.features) == point.label) 1 else 0
}.sum
// decision tree threshold needs to be specified
val dtTotalCorrect = data.map { point =>
  val score = dtModel.predict(point.features)
  val predicted = if (score > 0.5) 1 else 0 
  if (predicted == point.label) 1 else 0
}.sum
val svmAccuracy = svmTotalCorrect / numData
// svmAccuracy: Double = 0.5146720757268425
val nbAccuracy = nbTotalCorrect / numData
// nbAccuracy: Double = 0.5803921568627451
val dtAccuracy = dtTotalCorrect / numData
// dtAccuracy: Double = 0.6482758620689655

// compute area under PR and ROC curves for each model
// generate binary classification metrics
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
val metrics = Seq(lrModel, svmModel).map { model => 
	val scoreAndLabels = data.map { point =>
  		(model.predict(point.features), point.label)
	}
	val metrics = new BinaryClassificationMetrics(scoreAndLabels)
	(model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
}
// again, we need to use the special nbData for the naive Bayes metrics 
val nbMetrics = Seq(nbModel).map{ model =>
	val scoreAndLabels = nbData.map { point =>
  		val score = model.predict(point.features)
  		(if (score > 0.5) 1.0 else 0.0, point.label)
	}	
	val metrics = new BinaryClassificationMetrics(scoreAndLabels)
	(model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
}
// here we need to compute for decision tree separately since it does 
// not implement the ClassificationModel interface
val dtMetrics = Seq(dtModel).map{ model =>
	val scoreAndLabels = data.map { point =>
  		val score = model.predict(point.features)
  		(if (score > 0.5) 1.0 else 0.0, point.label)
	}	
	val metrics = new BinaryClassificationMetrics(scoreAndLabels)
	(model.getClass.getSimpleName, metrics.areaUnderPR, metrics.areaUnderROC)
}
val allMetrics = metrics ++ nbMetrics ++ dtMetrics
allMetrics.foreach{ case (m, pr, roc) => 
	println(f"$m, Area under PR: ${pr * 100.0}%2.4f%%, Area under ROC: ${roc * 100.0}%2.4f%%") 
}
/*
LogisticRegressionModel, Area under PR: 75.6759%, Area under ROC: 50.1418%
SVMModel, Area under PR: 75.6759%, Area under ROC: 50.1418%
NaiveBayesModel, Area under PR: 68.0851%, Area under ROC: 58.3559%
DecisionTreeModel, Area under PR: 74.3081%, Area under ROC: 64.8837%
*/

// standardizing the numerical features
import org.apache.spark.mllib.linalg.distributed.RowMatrix
val vectors = data.map(lp => lp.features)
val matrix = new RowMatrix(vectors)
val matrixSummary = matrix.computeColumnSummaryStatistics()

println(matrixSummary.mean)
// [0.41225805299526636,2.761823191986623,0.46823047328614004, ...
println(matrixSummary.min)
// [0.0,0.0,0.0,0.0,0.0,0.0,0.0,-1.0,0.0,0.0,0.0,0.045564223,-1.0, ...
println(matrixSummary.max)
// [0.999426,363.0,1.0,1.0,0.980392157,0.980392157,21.0,0.25,0.0,0.444444444, ...
println(matrixSummary.variance)
// [0.1097424416755897,74.30082476809638,0.04126316989120246, ...
println(matrixSummary.numNonzeros)
// [5053.0,7354.0,7172.0,6821.0,6160.0,5128.0,7350.0,1257.0,0.0,7362.0, ...

// scale the input features using MLlib's StandardScaler
import org.apache.spark.mllib.feature.StandardScaler
val scaler = new StandardScaler(withMean = true, withStd = true).fit(vectors)
val scaledData = data.map(lp => LabeledPoint(lp.label, scaler.transform(lp.features)))
// compare the raw features with the scaled features
println(data.first.features)
// [0.789131,2.055555556,0.676470588,0.205882353,
println(scaledData.first.features)
// [1.1376439023494747,-0.08193556218743517,1.025134766284205,-0.0558631837375738,
println((0.789131 - 0.41225805299526636)/math.sqrt(0.1097424416755897))
 // 1.137647336497682

// train a logistic regression model on the scaled data, and compute metrics
val lrModelScaled = LogisticRegressionWithSGD.train(scaledData, numIterations)
val lrTotalCorrectScaled = scaledData.map { point =>
  if (lrModelScaled.predict(point.features) == point.label) 1 else 0
}.sum
val lrAccuracyScaled = lrTotalCorrectScaled / numData
// lrAccuracyScaled: Double = 0.6204192021636241
val lrPredictionsVsTrue = scaledData.map { point => 
	(lrModelScaled.predict(point.features), point.label) 
}
val lrMetricsScaled = new BinaryClassificationMetrics(lrPredictionsVsTrue)
val lrPr = lrMetricsScaled.areaUnderPR
val lrRoc = lrMetricsScaled.areaUnderROC
println(f"${lrModelScaled.getClass.getSimpleName}\nAccuracy: ${lrAccuracyScaled * 100}%2.4f%%\nArea under PR: ${lrPr * 100.0}%2.4f%%\nArea under ROC: ${lrRoc * 100.0}%2.4f%%") 
/*
LogisticRegressionModel
Accuracy: 62.0419%
Area under PR: 72.7254%
Area under ROC: 61.9663%
*/

// Investigate the impact of adding in the 'category' feature
val categories = records.map(r => r(3)).distinct.collect.zipWithIndex.toMap
// categories: scala.collection.immutable.Map[String,Int] = Map("weather" -> 0, "sports" -> 6, 
//	"unknown" -> 4, "computer_internet" -> 12, "?" -> 11, "culture_politics" -> 3, "religion" -> 8,
// "recreation" -> 2, "arts_entertainment" -> 9, "health" -> 5, "law_crime" -> 10, "gaming" -> 13, 
// "business" -> 1, "science_technology" -> 7)
val numCategories = categories.size
// numCategories: Int = 14
val dataCategories = records.map { r =>
	val trimmed = r.map(_.replaceAll("\"", ""))
	val label = trimmed(r.size - 1).toInt
	val categoryIdx = categories(r(3))
	val categoryFeatures = Array.ofDim[Double](numCategories)
	categoryFeatures(categoryIdx) = 1.0
	val otherFeatures = trimmed.slice(4, r.size - 1).map(d => if (d == "?") 0.0 else d.toDouble)
	val features = categoryFeatures ++ otherFeatures
	LabeledPoint(label, Vectors.dense(features))
}
println(dataCategories.first)
// LabeledPoint(0.0, [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.789131,2.055555556,
//	0.676470588,0.205882353,0.047058824,0.023529412,0.443783175,0.0,0.0,0.09077381,0.0,0.245831182,
// 0.003883495,1.0,1.0,24.0,0.0,5424.0,170.0,8.0,0.152941176,0.079129575])


// standardize the feature vectors
val scalerCats = new StandardScaler(withMean = true, withStd = true).fit(dataCategories.map(lp => lp.features))
val scaledDataCats = dataCategories.map(lp => LabeledPoint(lp.label, scalerCats.transform(lp.features)))
println(dataCategories.first.features)
// [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.789131,2.055555556,0.676470588,0.205882353,
// 0.047058824,0.023529412,0.443783175,0.0,0.0,0.09077381,0.0,0.245831182,0.003883495,1.0,1.0,24.0,0.0,
// 5424.0,170.0,8.0,0.152941176,0.079129575]
println(scaledDataCats.first.features)
/*
[-0.023261105535492967,2.720728254208072,-0.4464200056407091,-0.2205258360869135,-0.028492999745483565,
-0.2709979963915644,-0.23272692307249684,-0.20165301179556835,-0.09914890962355712,-0.381812077600508,
-0.06487656833429316,-0.6807513271391559,-0.2041811690290381,-0.10189368073492189,1.1376439023494747,
-0.08193556218743517,1.0251347662842047,-0.0558631837375738,-0.4688883677664047,-0.35430044806743044
,-0.3175351615705111,0.3384496941616097,0.0,0.8288021759842215,-0.14726792180045598,0.22963544844991393,
-0.14162589530918376,0.7902364255801262,0.7171932152231301,-0.29799680188379124,-0.20346153667348232,
-0.03296720969318916,-0.0487811294839849,0.9400696843533806,-0.10869789547344721,-0.2788172632659348]
*/

// train model on scaled data and evaluate metrics
val lrModelScaledCats = LogisticRegressionWithSGD.train(scaledDataCats, numIterations)
val lrTotalCorrectScaledCats = scaledDataCats.map { point =>
  if (lrModelScaledCats.predict(point.features) == point.label) 1 else 0
}.sum
val lrAccuracyScaledCats = lrTotalCorrectScaledCats / numData
val lrPredictionsVsTrueCats = scaledDataCats.map { point => 
	(lrModelScaledCats.predict(point.features), point.label) 
}
val lrMetricsScaledCats = new BinaryClassificationMetrics(lrPredictionsVsTrueCats)
val lrPrCats = lrMetricsScaledCats.areaUnderPR
val lrRocCats = lrMetricsScaledCats.areaUnderROC
println(f"${lrModelScaledCats.getClass.getSimpleName}\nAccuracy: ${lrAccuracyScaledCats * 100}%2.4f%%\nArea under PR: ${lrPrCats * 100.0}%2.4f%%\nArea under ROC: ${lrRocCats * 100.0}%2.4f%%") 
/*
LogisticRegressionModel
Accuracy: 66.5720%
Area under PR: 75.7964%
Area under ROC: 66.5483%
*/

// train naive Bayes model with only categorical data
val dataNB = records.map { r =>
	val trimmed = r.map(_.replaceAll("\"", ""))
	val label = trimmed(r.size - 1).toInt
	val categoryIdx = categories(r(3))
	val categoryFeatures = Array.ofDim[Double](numCategories)
	categoryFeatures(categoryIdx) = 1.0
	LabeledPoint(label, Vectors.dense(categoryFeatures))
}
val nbModelCats = NaiveBayes.train(dataNB)
val nbTotalCorrectCats = dataNB.map { point =>
  if (nbModelCats.predict(point.features) == point.label) 1 else 0
}.sum
val nbAccuracyCats = nbTotalCorrectCats / numData
val nbPredictionsVsTrueCats = dataNB.map { point => 
	(nbModelCats.predict(point.features), point.label) 
}
val nbMetricsCats = new BinaryClassificationMetrics(nbPredictionsVsTrueCats)
val nbPrCats = nbMetricsCats.areaUnderPR
val nbRocCats = nbMetricsCats.areaUnderROC
println(f"${nbModelCats.getClass.getSimpleName}\nAccuracy: ${nbAccuracyCats * 100}%2.4f%%\nArea under PR: ${nbPrCats * 100.0}%2.4f%%\nArea under ROC: ${nbRocCats * 100.0}%2.4f%%") 
/*
NaiveBayesModel
Accuracy: 60.9601%
Area under PR: 74.0522%
Area under ROC: 60.5138%
*/

// investigate the impact of model parameters on performance
// create a training function
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.optimization.Updater
import org.apache.spark.mllib.optimization.SimpleUpdater
import org.apache.spark.mllib.optimization.L1Updater
import org.apache.spark.mllib.optimization.SquaredL2Updater
import org.apache.spark.mllib.classification.ClassificationModel

// helper function to train a logistic regresson model
def trainWithParams(input: RDD[LabeledPoint], regParam: Double, numIterations: Int, updater: Updater, stepSize: Double) = {
	val lr = new LogisticRegressionWithSGD
	lr.optimizer.setNumIterations(numIterations).setUpdater(updater).setRegParam(regParam).setStepSize(stepSize)
	lr.run(input)
}
// helper function to create AUC metric
def createMetrics(label: String, data: RDD[LabeledPoint], model: ClassificationModel) = {
	val scoreAndLabels = data.map { point =>
  		(model.predict(point.features), point.label)
	}
	val metrics = new BinaryClassificationMetrics(scoreAndLabels)
	(label, metrics.areaUnderROC)
}

// cache the data to increase speed of multiple runs agains the dataset
scaledDataCats.cache
// num iterations
val iterResults = Seq(1, 5, 10, 50).map { param =>
	val model = trainWithParams(scaledDataCats, 0.0, param, new SimpleUpdater, 1.0)
	createMetrics(s"$param iterations", scaledDataCats, model)
}
iterResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
/*
1 iterations, AUC = 64.97%
5 iterations, AUC = 66.62%
10 iterations, AUC = 66.55%
50 iterations, AUC = 66.81%
*/

// step size
val stepResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
	val model = trainWithParams(scaledDataCats, 0.0, numIterations, new SimpleUpdater, param)
	createMetrics(s"$param step size", scaledDataCats, model)
}
stepResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
/*
0.001 step size, AUC = 64.95%
0.01 step size, AUC = 65.00%
0.1 step size, AUC = 65.52%
1.0 step size, AUC = 66.55%
10.0 step size, AUC = 61.92%
*/

// regularization
val regResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
	val model = trainWithParams(scaledDataCats, param, numIterations, new SquaredL2Updater, 1.0)
	createMetrics(s"$param L2 regularization parameter", scaledDataCats, model)
}
regResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
/*
0.001 L2 regularization parameter, AUC = 66.55%
0.01 L2 regularization parameter, AUC = 66.55%
0.1 L2 regularization parameter, AUC = 66.63%
1.0 L2 regularization parameter, AUC = 66.04%
10.0 L2 regularization parameter, AUC = 35.33%
*/

// investigate decision tree
import org.apache.spark.mllib.tree.impurity.Impurity
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.mllib.tree.impurity.Gini
def trainDTWithParams(input: RDD[LabeledPoint], maxDepth: Int, impurity: Impurity) = {
	DecisionTree.train(input, Algo.Classification, impurity, maxDepth)
}
 
// investigate tree depth impact for Entropy impurity
val dtResultsEntropy = Seq(1, 2, 3, 4, 5, 10, 20).map { param =>
	val model = trainDTWithParams(data, param, Entropy)
	val scoreAndLabels = data.map { point =>
		val score = model.predict(point.features)
  		(if (score > 0.5) 1.0 else 0.0, point.label)
	}
	val metrics = new BinaryClassificationMetrics(scoreAndLabels)
	(s"$param tree depth", metrics.areaUnderROC)
}
dtResultsEntropy.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
/*
1 tree depth, AUC = 59.33%
2 tree depth, AUC = 61.68%
3 tree depth, AUC = 62.61%
4 tree depth, AUC = 63.63%
5 tree depth, AUC = 64.88%
10 tree depth, AUC = 76.26%
20 tree depth, AUC = 98.45%
*/

// investigate tree depth impact for Gini impurity
val dtResultsGini = Seq(1, 2, 3, 4, 5, 10, 20).map { param =>
	val model = trainDTWithParams(data, param, Gini)
	val scoreAndLabels = data.map { point =>
		val score = model.predict(point.features)
  		(if (score > 0.5) 1.0 else 0.0, point.label)
	}
	val metrics = new BinaryClassificationMetrics(scoreAndLabels)
	(s"$param tree depth", metrics.areaUnderROC)
}
dtResultsGini.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
/*
1 tree depth, AUC = 59.33%
2 tree depth, AUC = 61.68%
3 tree depth, AUC = 62.61%
4 tree depth, AUC = 63.63%
5 tree depth, AUC = 64.89%
10 tree depth, AUC = 78.37%
20 tree depth, AUC = 98.87%
*/

// investigate Naive Bayes parameters
def trainNBWithParams(input: RDD[LabeledPoint], lambda: Double) = {
	val nb = new NaiveBayes
	nb.setLambda(lambda)
	nb.run(input)
}
val nbResults = Seq(0.001, 0.01, 0.1, 1.0, 10.0).map { param =>
	val model = trainNBWithParams(dataNB, param)
	val scoreAndLabels = dataNB.map { point =>
  		(model.predict(point.features), point.label)
	}
	val metrics = new BinaryClassificationMetrics(scoreAndLabels)
	(s"$param lambda", metrics.areaUnderROC)
}
nbResults.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.2f%%") }
/*
0.001 lambda, AUC = 60.51%
0.01 lambda, AUC = 60.51%
0.1 lambda, AUC = 60.51%
1.0 lambda, AUC = 60.51%
10.0 lambda, AUC = 60.51%
*/

// illustrate cross-validation
// create a 60% / 40% train/test data split
val trainTestSplit = scaledDataCats.randomSplit(Array(0.6, 0.4), 123)
val train = trainTestSplit(0)
val test = trainTestSplit(1)
// now we train our model using the 'train' dataset, and compute predictions on unseen 'test' data
// in addition, we will evaluate the differing performance of regularization on training and test datasets
val regResultsTest = Seq(0.0, 0.001, 0.0025, 0.005, 0.01).map { param =>
	val model = trainWithParams(train, param, numIterations, new SquaredL2Updater, 1.0)
	createMetrics(s"$param L2 regularization parameter", test, model)
}
regResultsTest.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.6f%%") }
/*
0.0 L2 regularization parameter, AUC = 66.480874%
0.001 L2 regularization parameter, AUC = 66.480874%
0.0025 L2 regularization parameter, AUC = 66.515027%
0.005 L2 regularization parameter, AUC = 66.515027%
0.01 L2 regularization parameter, AUC = 66.549180%
*/

// training set results
val regResultsTrain = Seq(0.0, 0.001, 0.0025, 0.005, 0.01).map { param =>
	val model = trainWithParams(train, param, numIterations, new SquaredL2Updater, 1.0)
	createMetrics(s"$param L2 regularization parameter", train, model)
}
regResultsTrain.foreach { case (param, auc) => println(f"$param, AUC = ${auc * 100}%2.6f%%") }
/*
0.0 L2 regularization parameter, AUC = 66.260311%
0.001 L2 regularization parameter, AUC = 66.260311%
0.0025 L2 regularization parameter, AUC = 66.260311%
0.005 L2 regularization parameter, AUC = 66.238294%
0.01 L2 regularization parameter, AUC = 66.238294%
*/