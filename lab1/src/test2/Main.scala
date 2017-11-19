package test2

import userpackage._

import org.apache.spark.ml.feature.RegexTokenizer
import org.apache.spark.ml.Pipeline
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.sql.SQLContext

object Main {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("lab1").setMaster("local")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    import sqlContext.implicits._
    import sqlContext._

    val filePath = "src/test1/millionsong.txt"
    val rawDF = sparkContext.textFile(filePath).toDF("column")
    rawDF.show(5)
    
    //Step1: tokenize each row
    val regexTokenizer = new RegexTokenizer()
      .setInputCol("column")
      .setOutputCol("tokens")
      .setPattern(",")

    //Step2: transform with tokenizer and show 5 rows
    var tokenData = regexTokenizer.transform(rawDF)  //return a Dataset<Row>
    tokenData.take(5).foreach(println)

    //Step3: transform array of tokens to a vector of tokens (use our ArrayToVector)
     val arr2Vect = new Array2Vector().setInputCol("tokens").setOutputCol("features")
     var vectors = arr2Vect.transform(tokenData)
     println("vector of tokens are ")
     vectors.select("features").take(5).foreach(println)

    //Step4: extract the label(year) into a new column
     import org.apache.spark.ml.feature.VectorSlicer
     
     val lSlicer = new VectorSlicer().setInputCol("features").setOutputCol("yearVector")
     lSlicer.setIndices(Array(0))
     val data1 = lSlicer.transform(vectors)
     data1.select("yearVector").take(5).foreach(println)
     

    //Step5: convert type of the label from vector to double (use our Vector2Double)
     import org.apache.spark.ml.linalg.Vector
     val Invert: Vector => Double = _.apply(0)  //myDoubleUDF=>myDoubleUDF.apply(0)
     val v2d = new Vector2DoubleUDF(Invert)  //DoubleType
     
     v2d.setInputCol("yearVector").setOutputCol("yearDoubleUDF")
     val data2= v2d.transform(data1)
     data2.select("yearDoubleUDF").show(5)
      
    
    //Step6: shift all labels by the value of minimum label such that the value of the smallest becomes 0 (use our DoubleUDF) 
     val minYear : Double = 1922
     val newLable : Double => Double = OrgYear => {OrgYear- minYear}
     val lShifter = new DoubleUDF(newLable)
     lShifter.setInputCol("yearDoubleUDF").setOutputCol("newLabel")
     val data3=lShifter.transform(data2)
     data3.select("newLabel").show(5)
     
    //Step7: extract just the 3 first features in a new vector column
     val fSlicer = new VectorSlicer().setInputCol("features").setOutputCol("Top3features")
     fSlicer.setIndices(Array(0,1,2))
     val data4 = fSlicer.transform(data3)
     data4.select("newLabel","Top3features").show(5)
    
    //Step8: put everything together in a pipeline
     val pipeline = new Pipeline().setStages(Array(regexTokenizer, arr2Vect, lSlicer, v2d, lShifter, fSlicer))

    //Step9: generate model by fitting the rawDf into the pipeline
     val pipelineModel = pipeline.fit(rawDF)

    //Step10: transform data with the model - do predictions
      val testDF = sparkContext.textFile(filePath).toDF("column")
      var result = pipelineModel.transform(testDF)
      println("test result is  ")
      result.show(5)
      println("raw data result is ")
      testDF.show(5)

    //Step11: drop all columns from the dataframe other than label and features
      testDF.drop("column", "tokens","features", "yearVector", "yearDoubleUDF")
      
  }
}