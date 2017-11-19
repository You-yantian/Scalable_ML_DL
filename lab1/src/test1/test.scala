package test1

import scala.io.Source
import scala.Option
import scala.Some
import scala.None

object test {
  def main(args: Array[String]) {
    val processedDocuments = Source.fromFile("/d:/dataset/bathroom_bestwestern_hotel_sfo.1.gold").getLines.toIndexedSeq.zipWithIndex
    val documentShingles: Map[Int, Set[String]] = processedDocuments.map { document =>
    val shingles = document._1.toList.sliding(5)
        .map(_.mkString).toSet
        (document._2, shingles)
        }.toMap
   val randomHashFunctions = randomLinearHashFunction(50);
    
    val shingleVocab = documentShingles.values.flatten.toSet.toIndexedSeq.zipWithIndex.toMap
    println(processedDocuments)
  }
  
  def randomLinearHashFunction(n: Int) = {
    val slope = scala.util.Random.shuffle(0 to 1000);
    val const = scala.util.Random.shuffle(0 to 1000);
    slope.zip(const).take(50);
  }
}