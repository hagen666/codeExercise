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
 * 单个神经元，模拟神经网络的一个Cell，或者是线性回归问题
 *  forward output  = f(Wx+b)
 *  backward update : W,b
 *  backward output = delta(x)
 * */
public class Cell {
	private static AtomicInteger idgen = new AtomicInteger(1);
	//private INDArray wf;
	private INDArray wf;
	private INDArray bf;
	private ActiveFun fun;
	private LearningRate rate;
	private Optimizer optimizer;
	private GradientFields fields;
	private int gateId;
	public static final int UW = 0;
	public static final int UX = 1;
	public static final int UB = 2;
	private static final int UPDATE_SIZE = 3;
	
	public Cell(int vecLen,int biasLen,ActiveFun _fun,Optimizer _optimizer) {
		wf = Nd4j.rand(new int[] { biasLen, vecLen});
		bf = Nd4j.rand(new int[] {biasLen,1});
		fun = _fun;
		gateId = idgen.getAndAdd(1);
		optimizer = _optimizer;
		rate = optimizer.getLearningRate(UPDATE_SIZE, 0.1);
		fields = optimizer.getFields(UPDATE_SIZE);
	}
	public Cell(String filename,ActiveFun _fun,Optimizer _optimizer) throws IOException {
		FileInputStream fis = new FileInputStream(filename);
		DataInputStream dis = new DataInputStream(fis);
		wf = Nd4j.read(dis);
		bf = Nd4j.read(dis);
		dis.close();
		fun = _fun;
		gateId = idgen.getAndAdd(1);
		optimizer = _optimizer;
		rate = optimizer.getLearningRate(UPDATE_SIZE, 0.1);
		fields = optimizer.getFields(UPDATE_SIZE);
	}
	public void save(String filename) throws IOException {
		FileOutputStream fos = new FileOutputStream(filename);
		DataOutputStream dos = new DataOutputStream(fos);
		Nd4j.write(wf, dos);
		Nd4j.write(bf, dos);
		dos.flush();
		dos.close();
	}
	public String status() {
		return " u:"+wf+" b:"+bf;
	}
	public INDArray forward(INDArray x, INDArray _uf, INDArray _bf) {
		INDArray ufh = _uf.mmul(x);
		INDArray input = ufh.add(_bf);
		return fun.apply(input);
	}
	public INDArray forward(INDArray x) {
		return forward(x,wf,bf);
	}
	public Gradients getUpdateGradients(Gradients update) {
		update = optimizer.getDeltaGradient(fields, update, rate);
		return update;
	}
	public INDArray update(Gradients update) {
		Gradients g = optimizer.getDeltaGradient(fields, update, rate);
		return updateNoOptimizer(g);
	}
	public INDArray updateNoOptimizer(Gradients update) {
		wf.subi(update.delta[UW]);
		bf.subi(update.delta[UB]);
		return  update.delta[UX];
	}
	public Gradients backward(INDArray[] x, INDArray[] acumulateDiff) {
		Gradients u[] = new Gradients[x.length];
		for(int i = 0;i<x.length;i++) {
			u[i] = backward( x[i], acumulateDiff[i]);
		}
		return Abstract.getAverage(u);
	}
	public Gradients backward(INDArray x, INDArray acumulateDiff) {
		INDArray wfx = wf.mmul(x);
		INDArray input = wfx.add(x);
		INDArray diff = fun.diff(input);
		INDArray delta = null;
		if(acumulateDiff == null) {
			delta = diff;
		}else {
			delta = acumulateDiff.mul(diff);
		}
		INDArray uw = Util.getDeltaW(delta, x);
		INDArray ux = wf.transpose().mmul(delta);
		Gradients up = new Gradients(UPDATE_SIZE);
		up.delta[UB] = delta;
		up.delta[UX] = ux;
		up.delta[UW] = uw;
		return up;
	}
	public void printUpdater(Gradients u) {
		System.out.println(" uw:"+u.delta[UW]+" ub"+u.delta[UB]+ " ux "+u.delta[UX]);
	}
}
