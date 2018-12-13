package me.hagen.neural.fuc;

import org.nd4j.linalg.api.ndarray.INDArray;

public interface ActiveFun {

    public INDArray apply(INDArray input);

    public INDArray diff(INDArray diff);

}
