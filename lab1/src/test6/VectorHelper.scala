package test6

import org.apache.spark.ml.linalg.{Matrices, Vector, Vectors}
import scala.collection.immutable.VectorBuilder

object VectorHelper {
  def dot(v1: Vector, v2: Vector): Double = {
    
    if( v1.size != v2.size) 
        throw new IllegalArgumentException("Input vectors have different Size")
    var result:Double = 0
    
    for( i <- 0 to (v1.size -1 )){
      result += v1.apply(i) * v2.apply(i)
    }
    result
  }

  def dot(v: Vector, s: Double): Vector = {
    val array=v.toArray
    var result=Vectors.dense(array.map(x=>x*s))
    result
  }

  def sum(v1: Vector, v2: Vector): Vector = {
     if( v1.size != v2.size) 
        throw new IllegalArgumentException("Input vectors have different Size")
     var array1:Array[Double]=new Array(v1.size)
     for( i <- 0 to (v1.size -1 )){
      array1(i) = v1.apply(i) + v2.apply(i)
     }
     Vectors.dense(array1)
  }

  def fill(size: Int, fillVal: Double): Vector = {
     val array1:Array[Double]=new Array(size)
     for( i <- 0 to (size -1 )){
      array1(i) = fillVal
     }
     Vectors.dense(array1)
  }
}