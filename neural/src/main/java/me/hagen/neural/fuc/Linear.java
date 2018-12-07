package me.hagen.neural.fuc;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Linear implements ActiveFun {

	@Override
	public INDArray apply(INDArray input) {
		return input.dup();
	}

	@Override
	public INDArray diff(INDArray diff) {
		return Nd4j.ones(diff.shape());
	}

}
