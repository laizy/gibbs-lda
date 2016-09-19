#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include <random>
#include <ctime>
#include <iostream>
#include <cstring>
#include <string>


typedef struct _gibbs_lda_conf {
	double alpha;
	double beta;
	int T; // 总主题数
	unsigned int seed;
	int num_iter;
	int burnin;
} gibbs_lda_conf;

typedef struct _lda_input {
	int N; // 文档总词语数 
	int W; // # 词汇总数
	int D; // # 文档总数
	int *d; // size : N;
	int *w; // size : N;
} lda_input;

typedef struct _lda_result {
	int* wp;  // size : W * T; 词汇wi 在 主题tj 下的计数
	int* dp;	// size : D*T; 文档di 在 主题tj 下的计数
	int* ztot; // size: T;     主题tj 在所有词中的总计数
	int* z;    // size: N;   文档每个词赋予的主题
	int N; // 文档总词语数 
	int T; // 总主题数
	int W; // # 词汇总数
	int D; // # 文档总数

} lda_result;


int random_range(int l, int h)
{
	assert(h > l);
	int range = h - l;
	return l + rand() % range;
}
	
lda_result* lda_result_create(int N, int T, int W, int D);
void lda_result_destroy(lda_result * res);

void lda_result_print_summary(const lda_result* res);

int gibbs_sampler_lda(gibbs_lda_conf conf, const lda_input *in, lda_result* out )
{
	int T = conf.T;
	int N = in->N;
	int W = in->W;
	int *w = in->w;
	int *d = in->d;

	int* wp = out->wp;
	int* dp = out->dp;
	int* ztot = out->ztot;

	int* z = out->z;
	int* order = (int *)malloc(N*sizeof(int));
	double* topic_probs = (double *)malloc((1 + T) * sizeof(double));

	printf("start random initialization\n");

	std::default_random_engine generator(conf.seed++);
	std::uniform_int_distribution<int> int_random(0, T-1);// 注意该随机函数返回[a,b] 闭区间的数，包括b
	for	(int i=0; i < N; i++) {
		int topic = int_random(generator);
		assert(topic >= 0 && topic < T);
		z[i] = topic;
		int wi = w[i];
		int di = d[i];
		wp[wi*T + topic] ++;
		dp[di*T + topic] ++;
		ztot[topic] ++;
	}
	lda_result_print_summary(out);

	for	(int i=0; i< N; i++) order[i] = i;
	std::srand(unsigned(std::time(0)));
	std::random_shuffle(order, order + N);

	std::uniform_real_distribution<double> real_random(0.0, 1.0);

	for (int iter = 0, totiter = conf.num_iter; iter < totiter; iter++) {
		for (int ii	= 0; ii < N; ii++) {
			int i = order[ii];
			int wi = w[i], di = d[i], ti = z[i];

			int wioffset = wi*T;
			int dioffset = di*T;
			ztot[ti] --;
			wp[wioffset + ti] --;
			dp[dioffset + ti] --;

			double beta = conf.beta, alpha = conf.alpha;
			double wbeta = W*beta;
			topic_probs[0] = 0.0;
			for (int tj = 0; tj < T; tj++) {
				topic_probs[tj + 1] = topic_probs[tj] + (wp[wioffset + tj] + beta) / (ztot[tj] + wbeta) * (dp[dioffset + tj] + alpha);
			}

			double rand_prob = real_random(generator)*topic_probs[T];
			
			auto low = std::lower_bound(topic_probs, topic_probs + T + 1, rand_prob);
			ti = low - topic_probs - 1;
			assert(ti < T && ti >= 0);

			z[i] = ti;
			wp[wioffset + ti] ++;
			dp[dioffset + ti] ++;
			ztot[ti] ++;
		}
	}

	lda_result_print_summary(out);

	free(order);
	free(topic_probs);

	return 0;
}


lda_result* lda_result_create(int N, int T, int W, int D)
{
	lda_result* res = NULL;
	size_t size = sizeof(int)*(T + W*T + D*T + N );
	char * mem = (char *)malloc(size);
	memset(mem, 0, size);
	if (mem != NULL){
		res = (lda_result *)malloc(sizeof(lda_result));
		res->N = N;
		res->T = T;
		res->W = W;
		res->D = D;
		res->wp = (int *)(mem );
		res->dp = (int *)(mem + sizeof(int)*W*T);
		res->ztot = (int *)(mem + sizeof(int)*(W+D)*T);
		res->z    = (int *)(mem + sizeof(int) * (W + D + 1)*T);
	}

	return res;
}

void lda_result_print_summary(const lda_result* res)
{
	printf("\nprobs:\t");
	int N = res->N;
	int T = res->T;
	int W = res->W;
	int D = res->D;
	printf("\nwp:\t");
	for (int i=0; i< W*T; i++) {
		printf("%d\t", res->wp[i]);
	}

	printf("\ndp:\t");
	for (int i=0; i< D*T; i++) {
		printf("%d\t", res->dp[i]);
	}

	printf("\nztot:\t");
	for (int i=0; i< T; i++) {
		printf("%d\t", res->ztot[i]);
	}

	printf("\nz:\t");
	for (int i=0; i< N; i++) {
		printf("%d\t", res->z[i]);
	}
}

void lda_result_destroy(lda_result * res)
{
	free(res->wp);
	free(res);
}

#if MEM_CHECK
#include <crtdbg.h>
void set_memory_leak_detect()
{
	// Get current flag
	int flag = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);

	// Turn on leak-checking bit.
	flag |= _CRTDBG_LEAK_CHECK_DF;

	// Set flag to the new value.
	_CrtSetDbgFlag(flag);
}
#endif

void lda_driver(int seed)
{

#if MEM_CHECK
	set_memory_leak_detect();
#endif

	gibbs_lda_conf conf = {0.01, 0.001, 2, 3, 100, 100};
	lda_input input ;

	int words[12] = {1,2,3, 1,3,4, 2,4,5, 0,2,4 };
	int docs[12]  = {0,0,0, 1,1,1, 2,2,2, 3,3,3 };
	input.N = 12;
	input.W = 6;
	input.D = 4;
	input.d = docs;
	input.w = words;

	auto minmax_d = std::minmax_element(input.d, input.d + input.N);
	auto minmax_w = std::minmax_element(input.w, input.w + input.N);

	printf("minmax_d :\t %d\t %d \n", *minmax_d.first, *minmax_d.second);
	printf("minmax_w :\t %d\t %d \n", *minmax_w.first, *minmax_w.second);

	conf.seed = seed;

	lda_result* out = lda_result_create(input.N, conf.T, input.W, input.D);
	int info= gibbs_sampler_lda(conf, &input, out );

	lda_result_destroy(out);

}

void test_words()
{
	const char* a = "a b c d e aa bb a d c ee bb";

}

int main(int argc, char*argv[])
{
	lda_driver(argc);
	return 0;
}




