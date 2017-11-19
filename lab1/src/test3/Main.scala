package test3

import userpackage._
import org.apache.spark._
import org.apache.spark.sql.{SQLContext, DataFrame}
import org.apache.spark.ml.tuning.CrossValidatorModel
import org.apache.spark.ml.regression.{LinearRegression,LinearRegressionModel}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.RegexTokenizer

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/test1/millionsong.txt"
    //val obsDF: DataFrame = ???
    val obsDF = sparkContext.textFile(filePath).toDF("column")
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
    val newLable : Double => Double = OrgYear => {OrgYear- minYear}
    val lShifter = new DoubleUDF(newLable)
    lShifter.setInputCol("yearDoubleUDF").setOutputCol("label")
    
    val fSlicer = new VectorSlicer().setInputCol("features").setOutputCol("Top3features")
    fSlicer.setIndices(Array(0,1,2))
    
    
    val myLR = new LinearRegression().setMaxIter(10).setRegParam(0.1).setElasticNetParam(0.1).setLabelCol("label").setFeaturesCol("Top3features");
    val lrStage = Array(regexTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer, myLR)
    val task3pipeline = new Pipeline().setStages(lrStage)
    val task3pipelineModel: PipelineModel = task3pipeline.fit(obsDF)
    val lrModel = task3pipelineModel.stages(6).asInstanceOf[LinearRegressionModel]
    
    
    //print rmse of our model
     println("RSME of the model is "+lrModel.summary.rootMeanSquaredError)
    
    //do prediction - print first k
    val testDF = sparkContext.textFile(filePath).toDF("column")
    var predictions =  task3pipelineModel.transform(testDF)
    predictions.show(5)
  }
}