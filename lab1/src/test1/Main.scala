package test1

import se.kth.spark.lab1._

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
    //val rawDF = ???

    val rdd = sc.textFile(filePath)

    //Step1: print the first 5 rows, what is the delimiter, number of features and the data types?
    rdd.take(5).foreach(x=>{println(x)})
    
    //Step2: split each row into an array of features
    val recordsRdd = rdd.map(_.split(","))
    recordsRdd.take(5).foreach(x=>{println(x.deep.mkString("|"))})
   
  
    //Step3: map each row into a Song object by using the year label and the first three features  
    case class Song(date:Double, f1:Double, f2:Double, f3:Double){
      def printSong() : Unit = {
        println(s"year:$date feature1: $f1 feature2:$f2 feature3:$f3")
      }
    }
    
    val songsRdd = recordsRdd.map(x=>{Song(x(0).toDouble,x(1).toDouble,x(2).toDouble,x(3).toDouble)})
    songsRdd.take(5).foreach(x=>x.printSong())
    
    //Step4: convert your rdd into a datafram
    import org.apache.spark.sql.Row;
    import org.apache.spark.sql.types.{StructType,StructField,StringType};
    import org.apache.spark.sql.types.DoubleType
    
    val schema = StructType("date feature1 feature2 feature3".split(" ").map(x=>StructField(x, DoubleType, nullable = true)))
    val rowRDD = songsRdd.map(song=>Row(song.date, song.f1, song.f2, song.f3))
    val songsDf = sqlContext.createDataFrame(rowRDD, schema)
    songsDf.createOrReplaceTempView("songs")
    println( songsDf.head(5).deep)
    songsDf.printSchema()
    
    //Q1:How many songs are there in the DataFrame?
    println("Total number of songs is ")
    var result = sparkSession.sql("Select count(*) from songs ")
    result.show()
    
    //Q2:How many songs were released between the years 1998 and 2000?
    println("Total number of songs released between the years 1998 and 2000 is ")
    result = sparkSession.sql("Select count(*) from songs where date>1998 and date<=2000")
    result.show()
    
    //Q3:What is the min, max and mean value of the year column?
    println("The min value of the year column is ")
    result = sparkSession.sql("Select min(date) from songs ")
    result.show()
    println("The max value of the year column is ")
    result = sparkSession.sql("Select max(date) from songs ")
    result.show()
    println("The mean value of the year column is ")
    result = sparkSession.sql("Select mean(date) from songs ")
    result.show()
    
    //Q4:Show the number of songs per year between the years 2000 and 2010?
    println("Total number of songs released between the years 2000 and 2010 is ")
    result = sparkSession.sql("Select count(*) from songs where date>2000 and date<=2010")
    result.show()
    
    for(year<-2001 to 2010){
    println("Number of songs released in year "+year)   
    result = sparkSession.sql("Select count(*) from songs where date = "+year)
    result.show()    
    }
    
  }
}