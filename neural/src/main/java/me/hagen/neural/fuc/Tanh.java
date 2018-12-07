package me.hagen.neural.fuc;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Tanh implements ActiveFun {

	@Override
	public INDArray apply(INDArray input) {
		return Transforms.tanh(input);
	}

	@Override
	public INDArray diff(INDArray input) {
		INDArray output = apply(input);
		output = output.muli(output);
		output = output.subi(1);
		output = output.negi();
		return output;
	}

}
