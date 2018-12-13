package me.hagen.neural.unit;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicInteger;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import me.hagen.neural.Util;
import me.hagen.neural.fuc.ActiveFun;
import me.hagen.neural.optimize.Abstract;
import me.hagen.neural.optimize.Abstract.GradientFields;
import me.hagen.neural.optimize.Abstract.Gradients;
import me.hagen.neural.optimize.Abstract.Optimizer;
import me.hagen.neural.optimize.Abstract.LearningRate;

/**
 * 单个Gate，模拟LSTM的一个Gate
 * forward output  = f(Wx+Uh+b)
 * backward update : w,b,u
 * backward output = delta(h)
 */
public class Gate {
    private static AtomicInteger idgen = new AtomicInteger(1);
    private INDArray wf;
    private INDArray uf;
    private INDArray bf;
    private ActiveFun fun;
    private LearningRate rate;
    private Optimizer optimizer;
    private GradientFields fields;
    private int gateId;
    public static final int UW = 0;
    public static final int UU = 1;
    public static final int UH = 2;
    public static final int UB = 3;
    private static final int UPDATE_SIZE = 4;

    public Gate(int vecLen, int hiddenLen, int biasLen, ActiveFun _fun, Optimizer _optimizer) {
        wf = Nd4j.rand(new int[]{biasLen, vecLen});
        uf = Nd4j.rand(new int[]{biasLen, hiddenLen});
        bf = Nd4j.rand(new int[]{biasLen, 1});
        fun = _fun;
        gateId = idgen.getAndAdd(1);
        optimizer = _optimizer;
        rate = optimizer.getLearningRate(UPDATE_SIZE, 0.1);
        fields = optimizer.getFields(UPDATE_SIZE);
    }

    public Gate(String filename, ActiveFun _fun, Optimizer _optimizer) throws IOException {
        FileInputStream fis = new FileInputStream(filename);
        DataInputStream dis = new DataInputStream(fis);
        wf = Nd4j.read(dis);
        uf = Nd4j.read(dis);
        bf = Nd4j.read(dis);
        dis.close();
        fun = _fun;
        gateId = idgen.getAndAdd(1);
        optimizer = _optimizer;
        rate = optimizer.getLearningRate(UPDATE_SIZE, 0.1);
        fields = optimizer.getFields(UPDATE_SIZE);
    }

    public String status() {
        return "w:" + wf + " u:" + uf + " b:" + bf;
    }

    public void save(String filename) throws IOException {
        FileOutputStream fos = new FileOutputStream(filename);
        DataOutputStream dos = new DataOutputStream(fos);
        Nd4j.write(wf, dos);
        Nd4j.write(uf, dos);
        Nd4j.write(bf, dos);
        dos.flush();
        dos.close();
    }

    public INDArray forward(INDArray x, INDArray h, INDArray _wf, INDArray _uf, INDArray _bf) {
        INDArray wfx = _wf.mmul(x);
        INDArray ufh = _uf.mmul(h);
        INDArray input = wfx.add(ufh).add(_bf);
        return fun.apply(input);
    }

    public INDArray forward(INDArray x, INDArray h) {
        return forward(x, h, wf, uf, bf);
    }

    public Gradients getUpdateGradients(Gradients update) {
        update = optimizer.getDeltaGradient(fields, update, rate);
        return update;
    }

    public INDArray update(Gradients update) {
        Gradients g = getUpdateGradients(update);
        return updateNoOptimizer(g);
    }

    public INDArray updateNoOptimizer(Gradients update) {
        wf.subi(update.delta[UW]);
        uf.subi(update.delta[UU]);
        bf.subi(update.delta[UB]);
        return update.delta[UH];
    }

    public Gradients backward(INDArray[] x, INDArray[] h, INDArray[] acumulateDiff) {
        Gradients u[] = new Gradients[x.length];
        for (int i = 0; i < x.length; i++) {
            u[i] = backward(x[i], h[i], acumulateDiff[i]);
        }

        return Abstract.getAverage(u);
    }

    public Gradients backward(INDArray x, INDArray h, INDArray acumulateDiff) {
        INDArray wfx = wf.mmul(x);
        INDArray ufh = uf.mmul(h);
        INDArray input = wfx.add(ufh).add(h);
        INDArray diff = fun.diff(input);
        INDArray delta = null;
        if (acumulateDiff == null) {
            delta = diff;
        } else {
            delta = acumulateDiff.mul(diff);
        }

        INDArray uw = Util.getDeltaW(delta, x);
        INDArray uu = Util.getDeltaW(delta, h);
        INDArray uh = uf.transpose().mmul(delta);
        Gradients up = new Gradients(UPDATE_SIZE);
        up.delta[UB] = delta;
        up.delta[UH] = uh;
        up.delta[UU] = uu;
        up.delta[UW] = uw;
        return up;
		/*
		wf.addi(uw.mul(rate.wrate));
		uf.addi(uu.mul(rate.urate));
		bf.addi(delta.mul(rate.brate));
		return uh;*/
    }

    public void printUpdater(Gradients u) {
        System.out.println("uw:" + u.delta[UW] + " uu:" + u.delta[UU] + " ub" + u.delta[UB] + " uh " + u.delta[UH]);
    }
}
