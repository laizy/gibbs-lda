#include <stdio.h>
#include <stdlib.h> // rand
#include <assert.h>
#include <string.h>

typedef struct _gibbs_lda_conf {
	double alpha;
	double beta;
	int T; // 总主题数
	int burnup;
} gibbs_lda_conf;

typedef struct _lda_input {
	int N; // 文档总词语数 
	int W; // # 词汇总数
	int D; // # 文档总数
	int *d; // size : N;
	int *w; // size : N;
} lda_input;

typedef struct _lda_result {
	int* wp;  // size : W * T;
	int* dp;	// size : D*T;
	int* ztot; // size: T;
	double *probs; // size : N;
	int N; // 文档总词语数 
	int T; // 总主题数
	int W; // # 词汇总数

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

lda_result* gibbs_sampler_lda(gibbs_lda_conf conf, const lda_input *in )
{
	int T = conf.T;
	int N = in->N;
	int *w = in->w;
	int *d = in->d;
	lda_result* out = lda_result_create(in->N, conf.T, in->W, in->D);

	int* wp = out->wp;
	int* dp = out->dp;
	int* ztot = out->ztot;

	int* z = malloc(N*sizeof(int));

	printf("start random initialization\n");

	for	(int i=0; i < N; i++)
	{
		int topic = random_range(0, T);
		z[i] = topic;
		int wi = w[i];
		int di = d[i];
		wp[wi*T + topic] ++;
		dp[di*T + topic] ++;
		ztot[topic] ++;
	}
	lda_result_print_summary(out);

	printf("determine random order update sequence \n");


	return out;

}


lda_result* lda_result_create(int N, int T, int W, int D)
{
	lda_result* res = NULL;
	size_t size = sizeof(double)*N + sizeof(int)*(T + W*T + D*T);
	char * mem = malloc(size);
	memset(mem, 0, size);
	if (mem != NULL){
		res = malloc(sizeof(lda_result));
		res->N = N;
		res->T = T;
		res->W = W;
		res->probs = (double *)mem;
		res->wp = (int *)(mem + sizeof(double)*N);
		res->dp = (int *)(mem + sizeof(double)*N + sizeof(int)*W*T);
		res->ztot = (int *)(mem + sizeof(double)*N + sizeof(int)*(W+D)*T);

	}

	return res;
}

void lda_result_print_summary(const lda_result* res)
{
	printf("\nprobs:\t");
	int N = res->N;
	for (int i=0; i< N; i++)
	{
		printf("%lf\t", res->probs[i]);
	}
	printf("\nwp:\t");
	for (int i=0; i< N; i++)
	{
		printf("%d\t", res->wp[i]);
	}

	printf("\ndp:\t");
	for (int i=0; i< N; i++)
	{
		printf("%d\t", res->dp[i]);
	}

	printf("\nztot:\t");
	for (int i=0; i< res->T; i++)
	{
		printf("%d\t", res->ztot[i]);
	}

}

void lda_result_destroy(lda_result * res)
{
	free(res->probs);
	free(res);
}

int main()
{
	gibbs_lda_conf conf = {0.01, 0.001, 2, 100};
	lda_input input ;

	int words[12] = {1,2,3, 1,3,4, 2,4,5, 0,2,4 };
	int docs[12]  = {0,0,0, 1,1,1, 2,2,2, 3,3,3 };
	input.N = 12;
	input.W = 6;
	input.D = 4;
	input.d = docs;
	input.w = words;

	lda_result* output = gibbs_sampler_lda(conf, &input);

	lda_result_destroy(output);

    return 0;
}
