#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include <random>
#include <ctime>
#include <iostream>
#include <cstring>
#include <string>
#include <fstream>


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


	free(order);
	free(topic_probs);

	return 0;
}

// dp: size T, z :size N, w : size N
int gibbs_sampler_lda_predict(gibbs_lda_conf conf, const int *w, int N, const lda_result* out_pre, int* z, int *dp )
{
	int T = out_pre->T;
	int W = out_pre->T;

	int* ztot = out_pre->ztot;
	int* wp = out_pre->wp;

	int* order = (int *)malloc(N*sizeof(int));
	double* topic_probs = (double *)malloc((1 + T) * sizeof(double));

	std::default_random_engine generator(conf.seed++);
	std::uniform_int_distribution<int> int_random(0, T-1);// 注意该随机函数返回[a,b] 闭区间的数，包括b
	for	(int i=0; i < N; i++) {
		int topic = int_random(generator);
		assert(topic >= 0 && topic < T);
		z[i] = topic;
		dp[topic] ++;
	}

	for	(int i=0; i< N; i++) order[i] = i;
	std::srand(unsigned(std::time(0)));
	std::random_shuffle(order, order + N);

	std::uniform_real_distribution<double> real_random(0.0, 1.0);

	for (int iter = 0, totiter = conf.num_iter; iter < totiter; iter++) {
		for (int ii	= 0; ii < N; ii++) {
			int i = order[ii];
			int wi = w[i], ti = z[i];

			int wioffset = wi*T;
			ztot[ti] --;

			double beta = conf.beta, alpha = conf.alpha;
			double wbeta = W*beta;
			topic_probs[0] = 0.0;
			for (int tj = 0; tj < T; tj++) {
				topic_probs[tj + 1] = topic_probs[tj] + (wp[wioffset + tj] + beta) / (ztot[tj] + wbeta) * (dp[tj] + alpha);
			}

			double rand_prob = real_random(generator)*topic_probs[T];
			
			auto low = std::lower_bound(topic_probs, topic_probs + T + 1, rand_prob);
			ti = low - topic_probs - 1;
			assert(ti < T && ti >= 0);

			z[i] = ti;
			dp[ti] ++;
		}
	}

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

void lda_driver(int seed, int* words, int* docs, int N, int W, int D, std::vector<std::string> & cihui)
{

#if MEM_CHECK
	set_memory_leak_detect();
#endif

	gibbs_lda_conf conf = {0.01, 0.001, 2, 3, 100, 100};
	int T = 50;
	conf.T = T;
	lda_input input ;

	input.N = N;
	input.W = W;
	input.D = D;
	input.d = docs;
	input.w = words;

	auto minmax_d = std::minmax_element(input.d, input.d + input.N);
	auto minmax_w = std::minmax_element(input.w, input.w + input.N);

	printf("minmax_d :\t %d\t %d \n", *minmax_d.first, *minmax_d.second);
	printf("minmax_w :\t %d\t %d \n", *minmax_w.first, *minmax_w.second);

	conf.seed = seed;

	lda_result* out = lda_result_create(input.N, conf.T, input.W, input.D);
	auto time_start = std::time(0);
	int info= gibbs_sampler_lda(conf, &input, out );
	auto time_end = std::time(0);
	std::cout << "total time:" << time_end - time_start << "\n";

	std::vector<std::string> topic(conf.T);
	for (int ti = 0; ti < conf.T; ti++)
	{
		struct wordtopic {
			int wi;
			int ti;
			int count;
		};
		std::vector<wordtopic> wordtopics;
		for (int wi = 0; wi < W; wi++)
		{
			int count = out->wp[wi*T + ti];
			if (count != 0) {
				wordtopics.push_back({ wi, ti, count });
			}
		}
		std::sort(wordtopics.begin(), wordtopics.end(), 
			[](wordtopic a, wordtopic b) { 
				return a.count > b.count; 
		});

		if (wordtopics.size() > 15) {
			wordtopics.resize(15);
		}

		std::string str = "";
		std::for_each(wordtopics.begin(), wordtopics.end(), [&](wordtopic wt) {
			str += cihui[wt.wi] + "\t";
		});
		
		topic[ti] =str;
	}

	std::ofstream fout("result.txt", std::ios::out);
	std::for_each(topic.begin(), topic.end(), [&](std::string str) {
		fout << str.c_str() << std::endl;
	});
	fout.close();


	lda_result_destroy(out);

}

void test_words()
{
	const char* a = "a b c d e aa bb a d c ee bb";

}

int heap_test() {
	int myints[] = { 10,20,30,5,15 };
	auto comp = [](int a, int b) {return a > b; };
	std::make_heap(myints, myints + 5, comp);
	std::pop_heap(myints, myints + 5, comp);
	myints[4] = 90;
	std::push_heap(myints, myints + 5, comp);
	std::sort_heap(myints, myints + 5, comp);
	

	std::vector<int> v(myints, myints + 5);

	std::make_heap(v.begin(), v.end());
	std::cout << "initial max heap   : " << v.front() << '\n';

	std::pop_heap(v.begin(), v.end()); v.pop_back();
	std::cout << "max heap after pop : " << v.front() << '\n';

	v.push_back(99); std::push_heap(v.begin(), v.end());
	std::cout << "max heap after push: " << v.front() << '\n';

	std::sort_heap(v.begin(), v.end());

	std::cout << "final sorted range :";
	for (unsigned i = 0; i<v.size(); i++)
		std::cout << ' ' << v[i];

	std::cout << '\n';

	return 0;
}

void lda_test(int seed)
{
	std::ifstream fws("WS.txt", std::ios::in);
	std::ifstream fds("DS.txt", std::ios::in);
	int N = 0, N2;
	fws >> N;
	fds >> N2;
	assert(N == N2);
	std::vector<int> w(N);
	std::vector<int> d(N);
	for (int i = 0; i < N; i++) {
		fws >> w[i];
		w[i] --;
		fds >> d[i];
		d[i]--;
	}
	fws.close();
	fds.close();

	auto minmax_d = std::minmax_element(d.begin(), d.end());
	auto minmax_w = std::minmax_element(w.begin(), w.end());
	assert(*(minmax_d.first) == 0);
	assert(*(minmax_w.first) == 0);
	int W1 = *(minmax_w.second);
	int D = *(minmax_d.second);

	int W ;
	std::ifstream fwords("words.txt", std::ios::in);
	fwords >> W;

	std::vector<std::string> words(W);
	for (size_t i = 0; i < W; i++)
	{
		std::string str;

		fwords >> str;
		words[i] = str;
	}

	lda_driver(seed, &w[0], &d[0], N, W, D, words);
}


int main(int argc, char*argv[])
{
	lda_test(argc);
	//heap_test();

	return 0;
}




