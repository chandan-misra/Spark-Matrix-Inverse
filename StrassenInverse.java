
/*
 * Copyright (c) 1995, 2008, Oracle and/or its affiliates. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   - Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *
 *   - Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *   - Neither the name of Oracle or the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

package in.ac.iitkgp.atdc;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.Matrices;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.distributed.BlockMatrix;
import org.jblas.DoubleMatrix;
import org.jblas.Solve;

import scala.Tuple2;

/**
 * Class for inverting a positive definite square matrix. 
 */

public class StrassenInverse {
	
	/**
	 * Inverts this `BlockMatrix`. The matrix must be square and positive definite.
	 * 
	 * @param A The input matrix whose inversion is to be performed
	 * @param ctx The JavaSparkContext of the job 
	 * @param size The size of the matrix in terms of number of partitions. If the dimension
	 *        of the matrix is n and the dimension of each block is m, the value of size is
	 *        = n/m.
	 * @param blockSize The size of each block of the matrix.
	 * @return C of type [[BlockMatrix]] which is the inverted matrix of A.
	 */

	public static BlockMatrix inverse(BlockMatrix A, JavaSparkContext ctx, int size, int blockSize) {
		if (size == 1) {
			System.out.println("Leaf");
			final Broadcast<Integer> bblockSize = ctx.broadcast(blockSize);
			JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> inv_A = A.blocks().toJavaRDD().map(
					new Function<Tuple2<Tuple2<Object, Object>, Matrix>, Tuple2<Tuple2<Object, Object>, Matrix>>() {

						@Override
						public Tuple2<Tuple2<Object, Object>, Matrix> call(Tuple2<Tuple2<Object, Object>, Matrix> arg0)
								throws Exception {
							int blockSize = bblockSize.getValue();
							Tuple2<Tuple2<Object, Object>, Matrix> tuple = arg0;
							Tuple2<Object, Object> tuple2 = arg0._1;
							int rowIndex = tuple2._1$mcI$sp();
							int colIndex = tuple2._1$mcI$sp();
							Matrix matrix = arg0._2;
							DoubleMatrix mat = Solve.pinv(new DoubleMatrix(blockSize, blockSize, matrix.toArray()));
							matrix = Matrices.dense(matrix.numRows(), matrix.numCols(), mat.toArray());
							tuple = new Tuple2<Tuple2<Object, Object>, Matrix>(new Tuple2(rowIndex, colIndex), matrix);
							return tuple;
						}

					});

			BlockMatrix blockAInv = new BlockMatrix(inv_A.rdd(), blockSize, blockSize);
			return blockAInv;

		} else {
			System.out.println("Non-Leaf");
			size = size / 2;
			JavaPairRDD<String, Tuple2<Tuple2<Object, Object>, Matrix>> pairRDD = StrassenInverse.breakMat(A, ctx,
					size);
			BlockMatrix A11 = StrassenInverse._11(pairRDD, ctx, blockSize);
			BlockMatrix A12 = StrassenInverse._12(pairRDD, ctx, blockSize);
			BlockMatrix A21 = StrassenInverse._21(pairRDD, ctx, blockSize);
			BlockMatrix A22 = StrassenInverse._22(pairRDD, ctx, blockSize);
			BlockMatrix I = StrassenInverse.inverse(A11, ctx, size, blockSize);

			BlockMatrix II = A21.multiply(I);
			BlockMatrix III = I.multiply(A12);
			BlockMatrix IV = A21.multiply(III);
			BlockMatrix V = IV.subtract(A22);
			BlockMatrix VI = StrassenInverse.inverse(V, ctx, size, blockSize);
			BlockMatrix C12 = III.multiply(VI);
			BlockMatrix C21 = VI.multiply(II);
			BlockMatrix VII = III.multiply(C21);
			BlockMatrix C11 = I.multiply(VII);
			BlockMatrix C22 = StrassenInverse.scalerMul(ctx, VI, -1, blockSize);
			BlockMatrix C = StrassenInverse.reArrange(ctx, C11, C12, C21, C22, size, blockSize);
			return C;
		}
	}
	
	/**
	 * Arranges four sub-matrices of type [[BlockMatrix]] of size 2^(n-1) to a matrix of size
	 * of 2^(n) by changing the block indices appropriately. It returns a matrix  of type 
	 * [[BlockMatrix]] of size 2^(n) with all the blocks of this four sub-matrices.
	 * 
	 * @param ctx The JavaSparkContext of the job
	 * @param C11 The sub-matrix of type [[BlockMatrix]] in the upper-left portion
	 * @param C12 The sub-matrix of type [[BlockMatrix]] in the upper-right portion
	 * @param C21 The sub-matrix of type [[BlockMatrix]] in the lower-left portion
	 * @param C22 The sub-matrix of type [[BlockMatrix]] in the lower-right portion
	 * @param size size The size of the matrix in terms of number of partitions. If the dimension
	 *        of the matrix is n and the dimension of each block is m, the value of size is
	 *        = n/m.
	 * @param blockSize The size of each block of the matrix.
	 * @return Matrix C of type [[BlockMatrix]] with all the blocks of C11, C12, C21 and C22 with
	 *         their changed block indices.
	 */

	private static BlockMatrix reArrange(JavaSparkContext ctx, BlockMatrix C11, BlockMatrix C12, BlockMatrix C21,
			BlockMatrix C22, int size, int blockSize) {
		final Broadcast<Integer> bSize = ctx.broadcast(size);
		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> C11_RDD = C11.blocks().toJavaRDD();
		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> C12_RDD = C12.blocks().toJavaRDD();
		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> C21_RDD = C21.blocks().toJavaRDD();
		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> C22_RDD = C22.blocks().toJavaRDD();

		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> C12Arranged = C12_RDD
				.map(new Function<Tuple2<Tuple2<Object, Object>, Matrix>, Tuple2<Tuple2<Object, Object>, Matrix>>() {

					@Override
					public Tuple2<Tuple2<Object, Object>, Matrix> call(Tuple2<Tuple2<Object, Object>, Matrix> tuple)
							throws Exception {
						int size = bSize.getValue();
						int rowIndex = tuple._1._1$mcI$sp();
						int colIndex = tuple._1._2$mcI$sp() + size;
						Matrix matrix = tuple._2;
						return new Tuple2(new Tuple2(rowIndex, colIndex), matrix);
					}
				});

		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> C21Arranged = C21_RDD
				.map(new Function<Tuple2<Tuple2<Object, Object>, Matrix>, Tuple2<Tuple2<Object, Object>, Matrix>>() {

					@Override
					public Tuple2<Tuple2<Object, Object>, Matrix> call(Tuple2<Tuple2<Object, Object>, Matrix> tuple)
							throws Exception {
						int size = bSize.getValue();
						int rowIndex = tuple._1._1$mcI$sp() + size;
						int colIndex = tuple._1._2$mcI$sp();
						Matrix matrix = tuple._2;
						return new Tuple2(new Tuple2(rowIndex, colIndex), matrix);
					}
				});

		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> C22Arranged = C22_RDD
				.map(new Function<Tuple2<Tuple2<Object, Object>, Matrix>, Tuple2<Tuple2<Object, Object>, Matrix>>() {

					@Override
					public Tuple2<Tuple2<Object, Object>, Matrix> call(Tuple2<Tuple2<Object, Object>, Matrix> tuple)
							throws Exception {
						int size = bSize.getValue();
						int rowIndex = tuple._1._1$mcI$sp() + size;
						int colIndex = tuple._1._2$mcI$sp() + size;
						Matrix matrix = tuple._2;
						return new Tuple2(new Tuple2(rowIndex, colIndex), matrix);
					}
				});

		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> union = C11_RDD
				.union(C12Arranged.union(C21Arranged.union(C22Arranged)));
		BlockMatrix C = new BlockMatrix(union.rdd(), blockSize, blockSize);
		return C;
	}

	/**
	 * Multiplies each block of `A` with the `scalar`.
	 *  
	 * @param ctx The JavaSparkContext for the job.
	 * @param A The input matrix of type [[BlockMatrix]]
	 * @param scalar The double scalar with which the matrix, A, is to be multiplied.
	 * @param blockSize The block size of the matrix.
	 * @return Matrix `product` of type [[BlockMatrix]] 
	 */
	
	private static BlockMatrix scalerMul(JavaSparkContext ctx, BlockMatrix A, final double scalar, int blockSize) {
		final Broadcast<Integer> bblockSize = ctx.broadcast(blockSize);
		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> A_RDD = A.blocks().toJavaRDD();
		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> B_RDD = A_RDD
				.map(new Function<Tuple2<Tuple2<Object, Object>, Matrix>, Tuple2<Tuple2<Object, Object>, Matrix>>() {

					@Override
					public Tuple2<Tuple2<Object, Object>, Matrix> call(Tuple2<Tuple2<Object, Object>, Matrix> tuple)
							throws Exception {
						int blockSize = bblockSize.getValue();
						Tuple2<Tuple2<Object, Object>, Matrix> tuple2 = tuple;
						int rowIndex = tuple2._1._1$mcI$sp();
						int colIndex = tuple2._1._2$mcI$sp();
						Matrix matrix = tuple._2;
						DoubleMatrix candidate = new DoubleMatrix(matrix.toArray());
						DoubleMatrix product = candidate.muli(scalar);
						matrix = Matrices.dense(blockSize, blockSize, product.toArray());
						return new Tuple2(new Tuple2(rowIndex, colIndex), matrix);
					}
				});

		BlockMatrix product = new BlockMatrix(B_RDD.rdd(), blockSize, blockSize);
		return product;

	}
	
	/**
	 * Breaks the matrix of type [[BlockMatrix]] into four equal sized sub-matrices. Each block
	 * of each sub-matrix gets a tag or key and relative index inside that sub-matrix.
	 *  
	 * @param A The input matrix of type [[BlockMatrix]].
	 * @param ctx The JavaSparkContext of the job.
	 * @param size size size The size of the matrix in terms of number of partitions. If the dimension
	 *        of the matrix is n and the dimension of each block is m, the value of size is
	 *        = n/m.
	 * @return PairRDD `pairRDD` of [[<String, BlockMatrix.RDD>]] type. Each tuple consists of
	 *         a tag corresponds to block's coordinate and the RDD of blocks. 
	 */

	private static JavaPairRDD<String, Tuple2<Tuple2<Object, Object>, Matrix>> breakMat(BlockMatrix A,
			JavaSparkContext ctx, int size) {

		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> rdd = A.blocks().toJavaRDD();
		final Broadcast<Integer> bSize = ctx.broadcast(size);
		JavaPairRDD<String, Tuple2<Tuple2<Object, Object>, Matrix>> pairRDD = rdd.mapToPair(
				new PairFunction<Tuple2<Tuple2<Object, Object>, Matrix>, String, Tuple2<Tuple2<Object, Object>, Matrix>>() {

					@Override
					public Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> call(
							Tuple2<Tuple2<Object, Object>, Matrix> tuple) throws Exception {

						Tuple2<Object, Object> tuple1 = tuple._1;
						int rowIndex = tuple1._1$mcI$sp();
						int colIndex = tuple1._2$mcI$sp();
						Matrix matrix = tuple._2;
						String tag = "";
						if (rowIndex / bSize.value() == 0 && colIndex / bSize.value() == 0) {
							rowIndex = rowIndex % bSize.value();
							colIndex = colIndex % bSize.value();
							tag = "A11";
						} else if (rowIndex / bSize.value() == 0 && colIndex / bSize.value() == 1) {
							rowIndex = rowIndex % bSize.value();
							colIndex = colIndex % bSize.value();
							tag = "A12";
						} else if (rowIndex / bSize.value() == 1 && colIndex / bSize.value() == 0) {
							rowIndex = rowIndex % bSize.value();
							colIndex = colIndex % bSize.value();
							tag = "A21";
						} else {
							rowIndex = rowIndex % bSize.value();
							colIndex = colIndex % bSize.value();
							tag = "A22";
						}
						return new Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>>(tag,
								new Tuple2(new Tuple2(rowIndex, colIndex), matrix));
					}

				});

		return pairRDD;
	}
	
	/**
	 * Returns the upper-left sub-matrix of a broken matrix.
	 * 
	 * @param pairRDD The PairRDD of a broken RDD with a tag for each block. 
	 * @param ctx The JavaSparkContext for the job.
	 * @param blockSize The block size of the matrix.
	 * @return the upper-left sub-matrix of type [[BlockMatrix]] of a broken matrix.
	 */
	
	private static BlockMatrix _11(JavaPairRDD<String, Tuple2<Tuple2<Object, Object>, Matrix>> pairRDD,
			JavaSparkContext ctx, int blockSize) {
		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> rddEntry = pairRDD
				.filter(new Function<Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>>, Boolean>() {

					@Override
					public Boolean call(Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple) throws Exception {
						Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple2 = tuple;
						String tag = tuple2._1;
						return tag.equals("A11");
					}
				}).map(new Function<Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>>, Tuple2<Tuple2<Object, Object>, Matrix>>() {

					@Override
					public Tuple2<Tuple2<Object, Object>, Matrix> call(
							Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple) throws Exception {						
						return tuple._2;
					}
					
				});
		BlockMatrix matA = new BlockMatrix(rddEntry.rdd(), blockSize, blockSize);
		return matA;
	}
	
	/**
	 * Returns the upper-right sub-matrix of a broken matrix.
	 * 
	 * @param pairRDD The PairRDD of a broken RDD with a tag for each block. 
	 * @param ctx The JavaSparkContext for the job.
	 * @param blockSize The block size of the matrix.
	 * @return the upper-left sub-matrix of type [[BlockMatrix]] of a broken matrix.
	 */
	
	private static BlockMatrix _12(JavaPairRDD<String, Tuple2<Tuple2<Object, Object>, Matrix>> pairRDD,
			JavaSparkContext ctx, int blockSize) {
		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> rddEntry = pairRDD
				.filter(new Function<Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>>, Boolean>() {

					@Override
					public Boolean call(Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple) throws Exception {
						Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple2 = tuple;
						String tag = tuple2._1;
						return tag.equals("A12");
					}
				}).map(new Function<Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>>, Tuple2<Tuple2<Object, Object>, Matrix>>() {

					@Override
					public Tuple2<Tuple2<Object, Object>, Matrix> call(
							Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple) throws Exception {						
						return tuple._2;
					}
					
				});
		BlockMatrix matA = new BlockMatrix(rddEntry.rdd(), blockSize, blockSize);
		return matA;
	}
	
	/**
	 * Returns the lower-left sub-matrix of a broken matrix.
	 * 
	 * @param pairRDD The PairRDD of a broken RDD with a tag for each block. 
	 * @param ctx The JavaSparkContext for the job.
	 * @param blockSize The block size of the matrix.
	 * @return the upper-left sub-matrix of type [[BlockMatrix]] of a broken matrix.
	 */
	
	private static BlockMatrix _21(JavaPairRDD<String, Tuple2<Tuple2<Object, Object>, Matrix>> pairRDD,
			JavaSparkContext ctx, int blockSize) {
		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> rddEntry = pairRDD
				.filter(new Function<Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>>, Boolean>() {

					@Override
					public Boolean call(Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple) throws Exception {
						Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple2 = tuple;
						String tag = tuple2._1;
						return tag.equals("A21");
					}
				}).map(new Function<Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>>, Tuple2<Tuple2<Object, Object>, Matrix>>() {

					@Override
					public Tuple2<Tuple2<Object, Object>, Matrix> call(
							Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple) throws Exception {						
						return tuple._2;
					}
					
				});
		BlockMatrix matA = new BlockMatrix(rddEntry.rdd(), blockSize, blockSize);
		return matA;
	}
	
	/**
	 * Returns the lower-right sub-matrix of a broken matrix.
	 * 
	 * @param pairRDD The PairRDD of a broken RDD with a tag for each block. 
	 * @param ctx The JavaSparkContext for the job.
	 * @param blockSize The block size of the matrix.
	 * @return the upper-left sub-matrix of type [[BlockMatrix]] of a broken matrix.
	 */
	
	private static BlockMatrix _22(JavaPairRDD<String, Tuple2<Tuple2<Object, Object>, Matrix>> pairRDD,
			JavaSparkContext ctx, int blockSize) {
		JavaRDD<Tuple2<Tuple2<Object, Object>, Matrix>> rddEntry = pairRDD
				.filter(new Function<Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>>, Boolean>() {

					@Override
					public Boolean call(Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple) throws Exception {
						Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple2 = tuple;
						String tag = tuple2._1;
						return tag.equals("A22");
					}
				}).map(new Function<Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>>, Tuple2<Tuple2<Object, Object>, Matrix>>() {

					@Override
					public Tuple2<Tuple2<Object, Object>, Matrix> call(
							Tuple2<String, Tuple2<Tuple2<Object, Object>, Matrix>> tuple) throws Exception {						
						return tuple._2;
					}
					
				});
		BlockMatrix matA = new BlockMatrix(rddEntry.rdd(), blockSize, blockSize);
		return matA;
	}
}
