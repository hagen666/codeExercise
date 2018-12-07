昨晚突然萌生一个idea，然后不想用花花绿绿各种架子，写了那么多，还没实现idea，算了，近期懒得写了。
项目使用Maven，矩阵计算使用的Nd4j，其余的都是手写，真是边写边学，似乎也没学到啥。不过实现确实有点小复杂。
昨晚踌躇满志，今日混吃等死。

下图是一个神经元训练的demo
```java
    //随机生成数据集大小
		int setSize = 1000;
		//训练时每个batch大小
		int batchSize = 50;
		//训练轮次
		int batchTime = 10000;
		//输入向量维数
		int inputSize = 2;
		//输出维数
		int biasLen = 2;
		/**
		 * 训练数据准备
		 * */
		INDArray w = Nd4j.create(new double[][] {{0.7,0.5},{0.4,0.1}});
		INDArray b = Nd4j.create(new double[][] {{0.3},{0.5}});
		INDArray[] xset = new INDArray[setSize];
		INDArray[] yset = new INDArray[setSize];
		Cell g = new Cell(inputSize,biasLen,new Tanh(),new Momentum());
		for(int i =0;i<xset.length;i++) {
			xset[i]=Nd4j.rand(new int[] {inputSize,1});
			yset[i] = g.forward(xset[i],  w,  b);
		}
		/**
		 * 小批量获取训练数据，做反向传播
		 * */
		for(int i =0;i<batchTime;i++) {
			
			Set<Integer> iset = new HashSet<Integer>();
			/**
			 * Sample batchSize个[x,y] pairs.
			 * */
			while(iset.size()<batchSize) {
				int rand = (int)(Math.random()*setSize);
				if(rand>=setSize)rand--;
				iset.add(rand);
			}
			INDArray[] xsample = new INDArray[batchSize];
			INDArray[] ypredict = new INDArray[batchSize];
			INDArray[] ylabel = new INDArray[batchSize];
			INDArray[] acc = new INDArray[batchSize];
			int k = 0;
			double loss  = 0;
			for(Integer node: iset) {
				xsample[k]=xset[node];
				ypredict[k]=g.forward(xsample[k]);
				ylabel[k]=yset[node];
				acc[k]=ypredict[k].sub(ylabel[k]);
				loss += acc[k].norm2Number().doubleValue();
				k++;
			}
			loss/=k;
			/**
			 * 同时对一批数据做反向传播，获取该批数据的梯度均值
			 * */
			Gradients up = g.backward(xsample, acc);
			g.update(up);
			if(i%100==0) {
				System.out.println("loop i="+i);
				System.out.println("loss :"+loss);
				System.out.println(g.status());
			}
		}
```
