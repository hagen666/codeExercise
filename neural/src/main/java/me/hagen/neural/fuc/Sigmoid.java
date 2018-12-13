package me.hagen.neural.fuc;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Sigmoid implements ActiveFun {

    @Override
    public INDArray apply(INDArray input) {
        return Transforms.sigmoid(input);
    }

    @Override
    public INDArray diff(INDArray input) {
        INDArray origin = Transforms.sigmoid(input);
        INDArray another = origin.sub(1);
        another.negi();
        return origin.mul(another);
    }

}
