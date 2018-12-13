package me.hagen.neural;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Util {

    private Util() {
    }

    public static INDArray wxb(INDArray w, INDArray x, INDArray b) {
        INDArray mul = w.mmul(x);
        return mul.add(b);
    }

    public static INDArray wx(INDArray w, INDArray x) {
        INDArray mul = w.mmul(x);
        return mul;
    }

    public static INDArray sigmoid(INDArray input) {
        return Transforms.sigmoid(input);
    }

    public static INDArray tanh(INDArray input) {
        return Transforms.tanh(input);
    }

    public static INDArray relu(INDArray input) {
        return Transforms.relu(input);
    }

    public static INDArray getDeltaW(INDArray delta, INDArray x) {
        if (delta.shape()[0] == 1) {
            delta = delta.transpose();
        }
        long deltaLen = delta.shape()[0];
        if (x.shape()[0] != 1) {
            x = x.transpose();
        }
        long xlen = 1;
        if (x.shape().length > 1) {
            xlen = x.shape()[1];
        }
        delta = delta.broadcast(new long[]{deltaLen, xlen});
        x = x.broadcast(new long[]{deltaLen, xlen});
        return delta.mul(x);
    }
}
