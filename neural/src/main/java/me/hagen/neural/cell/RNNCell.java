package me.hagen.neural.cell;

import java.io.File;
import java.io.IOException;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

import me.hagen.neural.fuc.Linear;
import me.hagen.neural.fuc.Tanh;
import me.hagen.neural.optimize.Abstract;
import me.hagen.neural.optimize.Abstract.Gradients;
import me.hagen.neural.optimize.Abstract.Optimizer;
import me.hagen.neural.unit.Cell;
import me.hagen.neural.unit.Gate;

public class RNNCell {
    public static class RNNOutput {
        public INDArray output;
        public INDArray message;
    }

    private Gate gate;
    private Cell cell;
    private Optimizer optimizer;
    private int inputLen;
    private int hiddenLen;
    private int finalOutLen;

    public RNNCell(int _input, int _hidden, int _output, Optimizer optm) {
        optimizer = optm;
        inputLen = _input;
        hiddenLen = _hidden;
        finalOutLen = _output;
        gate = new Gate(inputLen, hiddenLen, hiddenLen, new Tanh(), optimizer);
        cell = new Cell(hiddenLen, finalOutLen, new Linear(), optimizer);
    }

    public RNNCell(int _input, int _hidden, int _output, Optimizer optm, String dir) throws IOException {
        optimizer = optm;
        inputLen = _input;
        hiddenLen = _hidden;
        finalOutLen = _output;
        gate = new Gate(dir + "/gate", new Tanh(), optimizer);
        cell = new Cell(dir + "/cell", new Linear(), optimizer);
    }

    public void save(String file) throws IOException {
        File f = new File(file);
        if (!f.exists()) f.mkdirs();
        gate.save(file + "/gate");
        cell.save(file + "/cell");
    }

    public RNNOutput forward(INDArray x, INDArray h) {
        INDArray out = gate.forward(x, h);
        INDArray output = cell.forward(out);
        RNNOutput rnnout = new RNNOutput();
        rnnout.output = Transforms.softmax(output);
        rnnout.message = out;
        return rnnout;
    }

    public INDArray updateCell(Gradients g1) {
        return cell.update(g1);
    }

    public Gradients backwardCell(INDArray x, INDArray h, INDArray outdiff) {
        INDArray out = gate.forward(x, h);
        Gradients g1 = cell.backward(out, outdiff);
        return g1;
    }

    public INDArray updateGate(Gradients g2) {
        return gate.update(g2);
    }

    public INDArray[] backward(INDArray[] x, INDArray[] h, INDArray[] outdiff, INDArray[] fromUpLayerDiff) {
        Gradients g1[] = new Gradients[x.length];
        INDArray gradiff[] = new INDArray[x.length];
        INDArray toDownLayerDiff[] = new INDArray[x.length];
        Gradients g2[] = new Gradients[x.length];
        for (int i = 0; i < x.length; i++) {
            g1[i] = backwardCell(x[i], h[i], outdiff[i]);
            gradiff[i] = g1[i].delta[Cell.UX];
        }
        Gradients avg1 = Abstract.getAverage(g1);
        INDArray gateDiff = cell.update(avg1);
        for (int i = 0; i < x.length; i++) {
            if (fromUpLayerDiff == null || fromUpLayerDiff[i] == null) {
                g2[i] = backwardGate(x[i], h[i], gradiff[i]);
            } else {
                INDArray diff = gradiff[i].mul(0.25).add(gateDiff).add(fromUpLayerDiff[i].mul(0.5));
                g2[i] = backwardGate(x[i], h[i], diff);
            }
            toDownLayerDiff[i] = g2[i].delta[Gate.UH];
        }
        Gradients avg2 = Abstract.getAverage(g2);
        INDArray avgDiff = gate.update(avg2);
        for (int i = 0; i < toDownLayerDiff.length; i++) {
            toDownLayerDiff[i] = toDownLayerDiff[i].muli(0.5).addi(avgDiff.mul(0.5));
        }
        return toDownLayerDiff;
    }

    public Gradients backwardGate(INDArray x, INDArray h, INDArray gradiff) {
        return gate.backward(x, h, gradiff);
    }
}
