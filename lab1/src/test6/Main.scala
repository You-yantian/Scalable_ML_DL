package test6

import org.apache.spark._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{PolynomialExpansion, RegexTokenizer, VectorAssembler, VectorSlicer}
import org.apache.spark.sql.{DataFrame, Row, SQLContext}
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.rdd.RDD
import se.kth.spark.lab1.{Array2Vector, DoubleUDF, Vector2DoubleUDF}

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local[6]")

    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/test1/millionsong.txt"
  
    val obsDF = sparkContext.textFile(filePath).toDF("column")
    ////Transformations from task 2///
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("column")
      .setOutputCol("tokens")
      .setPattern(",")
    
    val arr2Vect = new Array2Vector().setInputCol("tokens").setOutputCol("Allfeatures")
    
    import org.apache.spark.ml.feature.VectorSlicer
    val lSlicer = new VectorSlicer().setInputCol("Allfeatures").setOutputCol("yearVector")
    lSlicer.setIndices(Array(0))
    
    import org.apache.spark.ml.linalg.Vector
    val Invert: Vector => Double = _.apply(0)  //myDoubleUDF=>myDoubleUDF.apply(0)
    val v2d = new Vector2DoubleUDF(Invert)  //DoubleType 
    v2d.setInputCol("yearVector").setOutputCol("yearDoubleUDF")
    
    val minYear : Double = 1922
    val newLable : Double => Double = Year => {Year- minYear}
    val lShifter = new DoubleUDF(newLable)
    lShifter.setInputCol("yearDoubleUDF").setOutputCol("label")
    
    val fSlicer = new VectorSlicer().setInputCol("Allfeatures").setOutputCol("features")
    fSlicer.setIndices(Array(1,2,3,4,5,6,7,8,9,10))
    val myLR = new MyLinearRegressionImpl().setLabelCol("label").setFeaturesCol("features");

    var lrStage = Array(regexTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer, myLR)
    val pipeline = new Pipeline().setStages(lrStage)
    val pipelineModel:PipelineModel  = pipeline.fit(obsDF)
    val lrModel = pipelineModel.stages(6).asInstanceOf[MyLinearModelImpl]
    println("The rmse of each iteration ")
    lrModel.trainingError.foreach(println)

    val TestfilePath = "src/test1/test.txt"
    val testDF = sparkContext.textFile(TestfilePath).toDF("column")
    var predictions =  pipelineModel.transform(testDF)
    predictions.show(5)


  }
}