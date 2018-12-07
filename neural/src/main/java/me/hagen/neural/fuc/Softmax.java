package me.hagen.neural.fuc;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Softmax implements ActiveFun {

	@Override
	public INDArray apply(INDArray input) {
		return Transforms.softmax(input);
	}

	@Override
	public INDArray diff(INDArray diff) {
		// TODO 
		return null;
	}

}
