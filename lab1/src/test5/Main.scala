package test5

import userpackage._
import org.apache.spark._
import org.apache.spark.sql.{ SQLContext, DataFrame }
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.regression.{LinearRegression,LinearRegressionModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.PolynomialExpansion

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/test1/millionsong.txt"
    val obsDF: DataFrame = sparkContext.textFile(filePath).toDF("column")
    
    ////Transformations from task 2///
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("column")
      .setOutputCol("tokens")
      .setPattern(",")
    
    val arr2Vect = new Array2Vector().setInputCol("tokens").setOutputCol("features")
    
    import org.apache.spark.ml.feature.VectorSlicer
    val lSlicer = new VectorSlicer().setInputCol("features").setOutputCol("yearVector")
    lSlicer.setIndices(Array(0))
    
    import org.apache.spark.ml.linalg.Vector
    val Invert: Vector => Double = _.apply(0)  //myDoubleUDF=>myDoubleUDF.apply(0)
    val v2d = new Vector2DoubleUDF(Invert)  //DoubleType 
    v2d.setInputCol("yearVector").setOutputCol("yearDoubleUDF")
    
    val minYear : Double = 1922
    val newLable : Double => Double = Year => {Year- minYear}
    val lShifter = new DoubleUDF(newLable)
    lShifter.setInputCol("yearDoubleUDF").setOutputCol("label")
    
    val fSlicer = new VectorSlicer().setInputCol("features").setOutputCol("Top3features")
    fSlicer.setIndices(Array(0,1,2))
    
    val poli = new PolynomialExpansion().setInputCol("Top3features").setOutputCol("interFeature").setDegree(2)
    
    val myLR = new LinearRegression().setMaxIter(10).setRegParam(0.1).setElasticNetParam(0.1).setLabelCol("label").setFeaturesCol("interFeature");
    val lrStage = Array(regexTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer, poli, myLR)
    val pipeline = new Pipeline().setStages(lrStage)
    val pipelineModel: PipelineModel = pipeline.fit(obsDF)
    val lrModel = pipelineModel.stages(7).asInstanceOf[LinearRegressionModel]

    val paramGrid = new ParamGridBuilder().addGrid(myLR.maxIter, Array(3,5,10,20,50))
                                          .addGrid(myLR.regParam, Array(0.1, 0.2, 0.3,0.4,0.5))
                                          .build()
    val cv = new CrossValidator().setEstimator(pipeline)
                                 .setEvaluator(new RegressionEvaluator)
                                 .setEstimatorParamMaps(paramGrid)
                                 .setNumFolds(2)
                                 
    val cvModel: CrossValidatorModel = cv.fit(obsDF)
    val BestlrModel = cvModel.bestModel.asInstanceOf[PipelineModel].stages(7).asInstanceOf[LinearRegressionModel]
    //print rmse of our model
    println("Best maxItr: "+BestlrModel.getMaxIter)
    println("Best regParam: "+BestlrModel.getRegParam)
    println("rmse: "+ BestlrModel.summary.rootMeanSquaredError)
    
    //do prediction - print first k
    val TestfilePath = "src/test1/test.txt"
    val testDF = sparkContext.textFile(TestfilePath).toDF("column")
    var predictions =  cvModel.transform(testDF)
    predictions.show(5)
  }
}