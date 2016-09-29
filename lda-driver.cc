#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include <random>
#include <ctime>
#include <iostream>
#include <cstring>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_map>

#include <spdlog\spdlog.h>
#include <spdlog\fmt\fmt.h>


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

void report_progress(int current, int total, int width, const char * prefix) {
	std::string bar = prefix;
	if (current == total) {
		bar += "complete !";
		for (int i=0; i < width; i++) bar += " ";
		bar += "\n";
	}else {
		bar += "[";
		double percent = 100.0 *current / total;
		int i = 0;
		for (; i*total < current*width; i++) bar += "=";
		bar += fmt::format("{:.2f}%=>", percent);
		for (; i < width; i++) bar += "-";
		bar += "]\r";
	}

	std::cout << bar;
}


const char LDA_FILE_SIGN[10] = "LDA_MODAL";
const char LDA_FILE_PLACEHOLDER[10] = "LDA_HOLDE";

int lda_result_save(const lda_result* result, const char* file_name)
{
	int D = result->D, T = result->T, W = result->W, N = result->N;
	FILE* file = fopen(file_name, "wb");
	if (file == NULL) {
		std::cout << "can not open file:" << file_name << std::endl;
		return -1;
	}

	// 先在开始出占位，后面写完后进行覆盖
	int len = sizeof(LDA_FILE_SIGN) - 1;
	if (fwrite(LDA_FILE_PLACEHOLDER, len, 1, file) != 1 ) {
		std::cout << "write error file:"<<__FILE__<<"\tline:"<<__LINE__ <<std::endl;
		goto write_error;
	}

	int meta[4] = { D, N, W, T };
	if (fwrite(meta, sizeof(meta), 1, file) != 1 ) {
		std::cout << "write error file:"<<__FILE__<<"\tline:"<<__LINE__ <<std::endl;
		goto write_error;
	}

	if (fwrite(result->wp, sizeof(result->wp[0]), W *T, file) != W*T ) {
		std::cout << "write error file:"<<__FILE__<<"\tline:"<<__LINE__ <<std::endl;
		goto write_error;
	}

	if (fwrite(result->dp, sizeof(result->dp[0]), D *T, file) != D*T ) {
		std::cout << "write error file:"<<__FILE__<<"\tline:"<<__LINE__ <<std::endl;
		goto write_error;
	}

	if (fwrite(result->ztot, sizeof(result->ztot[0]), T, file) != T ) {
		std::cout << "write error file:"<<__FILE__<<"\tline:"<<__LINE__ <<std::endl;
		goto write_error;
	}

	if (fwrite(result->z, sizeof(result->z[0]), N, file) != N ) {
		std::cout << "write error file:"<<__FILE__<<"\tline:"<<__LINE__ <<std::endl;
		goto write_error;
	}

	rewind(file);  // 跳到文件开始处
	len = sizeof(LDA_FILE_SIGN) - 1;
	if (fwrite(LDA_FILE_SIGN, len, 1, file) != 1 ) {
		std::cout << "write error file:"<<__FILE__<<"\tline:"<<__LINE__ <<std::endl;
		goto write_error;
	}

	fclose(file);
	return 0;

write_error :
	fclose(file);
	return -1;
}

lda_result* lda_result_load( const char* file_name)
{
	lda_result* result = NULL;
	FILE* file = fopen(file_name, "rb");
	if (file == NULL) {
		std::cout << "can not open file:" << file_name << std::endl;
		return NULL;
	}

	const size_t len = sizeof(LDA_FILE_SIGN) - 1;
	char lda_file_sign[len + 1] = { '\0' };
	if (fread(lda_file_sign, len, 1, file) != 1) {
		std::cout << "read error file:" << __FILE__ << "\tline:" << __LINE__ << std::endl;
		goto read_error;
	}
	if (strncmp(LDA_FILE_SIGN, lda_file_sign, len) != 0) {
		std::cout << fmt::format(" file {} is not a valid lda model file \n", file_name);
		goto read_error;
	}

	int  meta[4] = { 0 };
	if (fread(meta, sizeof(int), 4, file) != 4) {
		std::cout << "read error file:" << __FILE__ << "\tline:" << __LINE__ << std::endl;
		goto read_error;
	}

	int D = meta[0], N = meta[1], W = meta[2], T = meta[3];

	result = lda_result_create(N, T, W, D);

	if (fread(result->wp, sizeof(result->wp[0]), W *T, file) != W*T) {
		std::cout << "read error file:" << __FILE__ << "\tline:" << __LINE__ << std::endl;
		goto read_error;
	}

	if (fread(result->dp, sizeof(result->dp[0]), D *T, file) != D*T) {
		std::cout << "read error file:" << __FILE__ << "\tline:" << __LINE__ << std::endl;
		goto read_error;
	}

	if (fread(result->ztot, sizeof(result->ztot[0]), T, file) != T) {
		std::cout << "read error file:" << __FILE__ << "\tline:" << __LINE__ << std::endl;
		goto read_error;
	}

	if (fread(result->z, sizeof(result->z[0]), N, file) != N) {
		std::cout << "read error file:" << __FILE__ << "\tline:" << __LINE__ << std::endl;
		goto read_error;
	}

	fclose(file);
	return result;

read_error:
	fclose(file);
	lda_result_destroy(result);
	return NULL;
}

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
		report_progress(iter, totiter, 80, "gibbs sampling ");

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

	report_progress(1, 1, 80, "gibbs sampling ");

	free(order);
	free(topic_probs);

	return 0;
}

// dp: size D*T, z :size N, w : size N, d : size N
int gibbs_sampler_lda_predict(gibbs_lda_conf conf, const int *w, const int *d, 
	int N, int D,  const lda_result* out_pre, int* z, int *dp )
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
		dp[d[i] * T + topic] ++;
	}

	for	(int i=0; i< N; i++) order[i] = i;
	std::srand(unsigned(std::time(0)));
	std::random_shuffle(order, order + N);

	std::uniform_real_distribution<double> real_random(0.0, 1.0);

	double beta = conf.beta, alpha = conf.alpha;
	double wbeta = W*beta;
	for (int iter = 0, totiter = conf.num_iter; iter < totiter; iter++) {
		report_progress(iter, totiter, 80, "gibbs sampling ");

		for (int ii	= 0; ii < N; ii++) {
			int i = order[ii];
			int wi = w[i], ti = z[i], di = d[i];

			int wioffset = wi*T;
			int dioffset = di*T;
			dp[dioffset + ti] --;
			//ztot[ti] --;

			topic_probs[0] = 0.0;
			for (int tj = 0; tj < T; tj++) {
				topic_probs[tj + 1] = topic_probs[tj] + (wp[wioffset + tj] + beta) / (ztot[tj] + wbeta) * (dp[tj] + alpha);
			}

			double rand_prob = real_random(generator)*topic_probs[T];
			
			auto low = std::lower_bound(topic_probs, topic_probs + T + 1, rand_prob);
			ti = low - topic_probs - 1;
			assert(ti < T && ti >= 0);

			z[i] = ti;
			dp[dioffset + ti] ++;
			//ztot[ti] ++;
		}
	}

	report_progress(1, 1, 80, "gibbs sampling ");

	free(order);
	free(topic_probs);

	return 0;
}


lda_result* lda_result_create(int N, int T, int W, int D)
{
	lda_result* res = (lda_result *)malloc(sizeof(lda_result));
	if (res == NULL) {
		goto error;
	}
	size_t size = sizeof(int)*(T + W*T + D*T + N );
	char * mem = (char *)malloc(size);
	if (mem == NULL) {
		goto error;
	}
	memset(mem, 0, size);
	res->N = N;
	res->T = T;
	res->W = W;
	res->D = D;
	res->wp = (int *)(mem );
	res->dp = (int *)(mem + sizeof(int)*W*T);
	res->ztot = (int *)(mem + sizeof(int)*(W+D)*T);
	res->z    = (int *)(mem + sizeof(int) * (W + D + 1)*T);

	return res;
error:
	free(mem);
	free(res);
	return NULL;
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
	if (res) {
		free(res->wp);
		free(res);
	}
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

void lda_driver(int seed, int* words, int* docs, int N, int W, int D, 
	std::vector<std::string> & cihui, std::string& out_name)
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

	lda_result_save(out, out_name.c_str());

	std::vector<std::string> topic(conf.T);
	for (int ti = 0; ti < conf.T; ti++) {
		struct wordtopic {
			int wi;
			int ti;
			int count;
		};
		std::vector<wordtopic> wordtopics;
		for (int wi = 0; wi < W; wi++) {
			int count = out->wp[wi*T + ti];
			if (count != 0) {
				wordtopics.push_back({ wi, ti, count });
			}
		}
		std::sort(wordtopics.begin(), wordtopics.end(), 
			[](wordtopic a, wordtopic b) { 
				return a.count > b.count; 
		});

		if (wordtopics.size() > 25) {
			wordtopics.resize(25);
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

double divergence_KL(int*w1, int* w2, int N)
{
	int sum1 = 0, sum2 = 0;
	double KL12 = 0.0, KL21 = 0.0;
	for (int i = 0; i < N; i++) {
		KL12 += w1[i] * log2((w1[i] + 1e-5) / (w2[i] + 1e-5));
		KL21 += w2[i] * log2((w2[i] + 1e-5) / (w1[i] + 1e-5));
		sum1 += w1[i];
		sum2 += w2[i];
	}
	return 0.5*(KL12 / sum1 + KL21 / sum2);
}

void lda_words_similary(const char * model_name )
{

#if MEM_CHECK
	set_memory_leak_detect();
#endif

	lda_result* out = lda_result_load(model_name);
	int T = out->T;
	int N = out->N;
	int D = out->D;
	int W;
	std::ifstream fwords("words.txt", std::ios::in);
	fwords >> W;

	assert(W == out->W);

	std::vector<std::string> cihui(W);
	for (size_t i = 0; i < W; i++) {
		std::string str;
		fwords >> str;
		cihui[i] = str;
	}

	const int NP = 10;
	struct word_pair {
		int wi;
		int wj;
		double skl;  // 对称KL 散度
	};

	std::vector<word_pair> wordpairs;
	wordpairs.reserve(NP + 1);
	auto word_pair_comp = [](word_pair a, word_pair b) {
		return a.skl < b.skl;
	};

	std::ofstream fsimi("word_similarity.txt", std::ios::out);
	for (int wi = 0; wi < W; wi++) {
		wordpairs.clear();
		for (int wj = 0; wj < W; wj++) {
			if (wi == wj) continue;

			double skl = divergence_KL(&(out->wp[wi*T]), &(out->wp[wj*T]), T);
			if (wordpairs.size() < NP) {
				wordpairs.push_back({ wi, wj, skl });
				std::push_heap(wordpairs.begin(), wordpairs.end(), word_pair_comp);
			} else if (skl < wordpairs[0].skl) {
				wordpairs.push_back({ wi, wj, skl });
				std::push_heap(wordpairs.begin(), wordpairs.end(), word_pair_comp);
				std::pop_heap(wordpairs.begin(), wordpairs.end(), word_pair_comp);
				wordpairs.pop_back();
			}
		}

		std::sort_heap(wordpairs.begin(), wordpairs.end(), word_pair_comp);
		std::ostringstream str;
		str << cihui[wi] << "\t";
		std::for_each(wordpairs.begin(), wordpairs.end(), [&](word_pair wp) {
			str << cihui[wp.wj] << ":" << wp.skl << "\t";
		});
		str << "\n";
		fsimi << str.str();
	}

	fsimi.close();

	lda_result_destroy(out);

}


//注意：当字符串为空时，也会返回一个空字符串  
void split(std::string& s, char delim, std::vector< std::string >* ret)
{
	size_t last = 0;
	size_t index = s.find_first_of(delim, last);
	while (index != std::string::npos) {
		ret->push_back(s.substr(last, index - last));
		last = index + 1;
		index = s.find_first_of(delim, last);
	}
	if (index - last>0) {
		ret->push_back(s.substr(last, index - last));
	}
}

void process_text()
{
	using namespace std;

	ifstream fdocs("docs.txt");
	ofstream fwords("words.txt");
	ofstream fds("ds.txt");
	ofstream fws("ws.txt");
	string line;

	std::unordered_map<std::string, int> words_map;

	std::vector<std::string> docswords;
	int W = 0;
	int ntokens = 0;
	int D = 0;
	std::ostringstream ds;
	std::ostringstream ws;
	std::ostringstream words;
	while ( getline(fdocs, line) ) {
		split(line, '\t', &docswords);
		if (docswords.size() > 0) {
			std::for_each(docswords.begin(), docswords.end(), [&](std::string word) {
				if (words_map.find(word) == words_map.end()) {
					words_map[word] = W++;
					words << word << "\n";
				}
				
				ntokens ++;
				ws << words_map[word] << "\n";
				ds << D << "\n";
			});
			D++;
		}
		docswords.clear();
	}

	fds << ntokens << "\n" << ds.str();
	fws << ntokens << "\n" << ws.str();
	fwords << W << "\n" << words.str();
	fdocs.close();
	fds.close();
	fws.close();
}

void process_text2(std::string& docs_file, std::string& out_file)
{
	using namespace std;

	ifstream fdocs(docs_file, std::ios::in);
	ofstream fwords(out_file + ".words.txt");
	ofstream fds(out_file + ".ds.txt");
	ofstream fws(out_file + ".ws.txt");
	string line;

	std::unordered_map<std::string, int> words_map;

	std::vector<std::string> docswords;
	int W = 0;
	int ntokens = 0;
	int D = 0;
	std::ostringstream ds;
	std::ostringstream ws;
	std::ostringstream words;
	while (getline(fdocs, line)) {
		split(line, '\t', &docswords);
		if (docswords.size() > 0) {
			std::for_each(docswords.begin(), docswords.end(), [&](std::string word) {
				if (words_map.find(word) == words_map.end()) {
					words_map[word] = W++;
					words << word << "\n";
				}

				ntokens++;
				ws << words_map[word] << "\n";
				ds << D << "\n";
			});
			D++;
		}
		docswords.clear();
	}

	fds << ntokens << "\n" << ds.str();
	fws << ntokens << "\n" << ws.str();
	fwords << W << "\n" << words.str();
	fdocs.close();
	fds.close();
	fws.close();
}

// words 以\n 分隔
int lda_corpus_load(const char* corpus_name, lda_input& input, char * *words, size_t wdsize)
{
	// 文件格式: LDA_CORPUS + 文档数D(int 4 字节) + 总词汇数W + 词汇列表的字节总数wdsize
	//  + (D+1)个int(每个文档开始的偏移值, 每个文档的范围为 [di, di+1) d0 = 0  ) + 每个文档的具体词语编号（长度为所有文档词的总和) 
	//  + 词汇列表(以\n 分隔))
	const char LDA_CORPUS[11] = "LDA_CORPUS";
	FILE* fcorpus = fopen(corpus_name, "wb");
	if (fcorpus == NULL) {
		return -1;
	}

	char lda_corpus[10] = { 0 };
	if (fread(lda_corpus, sizeof(lda_corpus), 1, fcorpus) != 1) {
		goto io_error;
	}
	if (strncmp(LDA_CORPUS, lda_corpus, 10) != 0) {
		std::cout << "not a valid lda corpus file\n";
		return -1;
	}

	int D, W, N, wdsize;
	int * ws = NULL;
	int * ds = NULL;
	char * wordstr = NULL;

	if (fread(&D, sizeof(D), 1, fcorpus) != 1) {
		goto io_error;
	}
	if (fread(&W, sizeof(W), 1, fcorpus) != 1) {
		goto io_error;
	}
	if (fread(&wdsize, sizeof(wdsize), 1, fcorpus) != 1) {
		goto io_error;
	}

	std::vector<int> doffset(D + 1);
	if (fread(&doffset[0], sizeof(int)*(D + 1), 1, fcorpus) != 1) {
		goto io_error;
	}

	N = doffset[D];
	ds = (int *)malloc(sizeof(int) * N);
	ws = (int *)malloc(sizeof(int) * N);
	wordstr = (char *)malloc(sizeof(char)*wdsize);

	if (ds == NULL || ws == NULL || wordstr) {
		goto io_error;
	}

	for (int i = 0; i < D; i++) {
		for (int p = doffset[i], q = doffset[i+1]; p < q; p++) ds[p] = i;
	}

	if (fread(ws, sizeof(int)*N , 1, fcorpus) != 1) {
		goto io_error;
	}

	if (fread(wordstr, sizeof(char)*wdsize , 1, fcorpus) != 1) {
		goto io_error;
	}

	input.N = N;
	input.W = W;
	input.D = D;
	input.d = ds;
	input.w = ws;

	*words = wordstr;

	return 0;

io_error:
	fclose(fcorpus);
	free(ws);
	free(ds);
	free(wordstr);

	return -1;
}

// words 以\n 分隔
int lda_corpus_save(const char* corpus_name, int D, int W, int N, int *ds, int * ws, const char * words, size_t wdsize)
{
	// 文件格式: LDA_CORPUS + 文档数D(int 4 字节) + 总词汇数W + 词汇列表的字节总数wdsize
	//  + (D+1)个int(每个文档开始的偏移值, 每个文档的范围为 [di, di+1) d0 = 0  ) + 每个文档的具体词语编号（长度为所有文档词的总和) 
	//  + 词汇列表(以\n 分隔)
	const char LDA_CORPUS[11] = "LDA_CORPUS";
	FILE* fcorpus = fopen(corpus_name, "wb");

	if (fseek(fcorpus, sizeof(LDA_CORPUS) - 1, SEEK_SET) != 0) {
		goto io_error;
	}
	if (fwrite(&D, sizeof(D), 1, fcorpus) != 1) {
		goto io_error;
	}
	if (fwrite(&W, sizeof(W), 1, fcorpus) != 1) {
		goto io_error;
	}
	if (fwrite(&wdsize, sizeof(wdsize), 1, fcorpus) != 1) {
		goto io_error;
	}
	if (fwrite(ds, sizeof(ds), D + 1, fcorpus) != D + 1) {
		goto io_error;
	}
	if (fwrite(ws, sizeof(ws), N, fcorpus) != N) {
		goto io_error;
	}
	if (fwrite(words, wdsize, 1, fcorpus) != 1) {
		goto io_error;
	}

	rewind(fcorpus);
	if (fwrite(LDA_CORPUS, sizeof(LDA_CORPUS) - 1, 1, fcorpus) != 1) {
		goto io_error;
	}

	fclose(fcorpus);
	return 0;

io_error:
	fclose(fcorpus);
	return -1;
}

// 还没开发完.
int process_text3(std::string& docs_file, std::string& out_file)
{
	using namespace std;

	ifstream fdocs(docs_file, std::ios::in);
	
	std::unordered_map<std::string, int> words_map;

	std::vector<std::string> docswords;
	int W = 0;
	std::vector<int> ws;
	std::ostringstream words;
	std::vector<int> ds(1, 0);
	size_t dpos = 0;
	string line;
	while (getline(fdocs, line)) {
		split(line, '\t', &docswords);
		if (docswords.size() > 0) {
			std::for_each(docswords.begin(), docswords.end(), [&](std::string word) {
				auto wordpos = words_map.find(word);
				int wi = -1;
				if (wordpos == words_map.end()) {
					words_map[word] = wi = W++;
					words << word << "\n";
				} else {
					wi = wordpos->second;
				}
				ws.push_back(wi);
			});
			dpos += docswords.size();
			ds.push_back(dpos);
		}
		docswords.clear();
	}

	fdocs.close();

	int N = ws.size();
	assert(dpos == N);
	int D = ds.size() - 1;
	auto wstr = words.str();

	return lda_corpus_save(out_file.c_str(), D, W, N, &ds[0], &ws[0], wstr.c_str(), wstr.size());
}

void lda_train(int seed, std::string& corpus_name, std::string& out_name)
{
	std::ifstream fws(corpus_name + ".ws.txt", std::ios::in);
	std::ifstream fds(corpus_name + ".ds.txt", std::ios::in);
	int N = 0, N2;
	fws >> N;
	fds >> N2;
	assert(N == N2);
	std::vector<int> w(N);
	std::vector<int> d(N);
	for (int i = 0; i < N; i++) {
		fws >> w[i];
		fds >> d[i];
	}

	fws.close();
	fds.close();

	auto minmax_d = std::minmax_element(d.begin(), d.end());
	auto minmax_w = std::minmax_element(w.begin(), w.end());
	assert(*(minmax_d.first) == 0);
	assert(*(minmax_w.first) == 0);
	int W = *(minmax_w.second);
	int D = *(minmax_d.second);

	int T = 50;
	gibbs_lda_conf conf = { 0.01, 0.001, 2, 3, 100, 100 };
	conf.T = T;
	conf.seed = seed;

	lda_input input;
	input.N = N;
	input.W = W;
	input.D = D;
	input.d = &d[0];
	input.w = &w[0];

	lda_corpus_load(&input);

	lda_result* out = lda_result_create(input.N, conf.T, input.W, input.D);
	auto time_start = std::time(0);
	int info = gibbs_sampler_lda(conf, &input, out);
	auto time_end = std::time(0);
	std::cout << "total time:" << time_end - time_start << "\n";

	lda_result_save(out, out_name.c_str());

	lda_result_destroy(out);

}

void lda_predict(int seed, std::string& model_name, std::string& docs_name)
{
	lda_result* model = lda_result_load(model_name.c_str());
	int W = 0;
	std::ifstream fwords("corpus.words.txt", std::ios::in);
	fwords >> W;

	std::unordered_map<std::string, int> words_map;
	for (size_t i = 0; i < W; i++) {
		std::string str;
		fwords >> str;
		words_map[str] = i;
	}

	using namespace std;
	ifstream fdocs(docs_name, std::ios::in);
	ofstream fout(docs_name + ".predict.txt");
	string line;

	std::vector<std::string> docswords;
	int D = 0;
	std::ostringstream words;
	std::vector<int> ws;
	std::vector<int> ds;
	while (getline(fdocs, line)) {
		split(line, '\t', &docswords);
		if (docswords.size() > 0) {
			std::for_each(docswords.begin(), docswords.end(), [&](std::string word) {
				// 如果训练词汇表里没有该词语,则忽略
				auto wdpair = words_map.find(word);
				if (wdpair != words_map.end()) {
					ws.push_back(wdpair->second);
					ds.push_back(D);
				}
			});
			D++;
		}
		docswords.clear();
	}

	fdocs.close();

	int T = 50;
	gibbs_lda_conf conf = { 0.01, 0.001, 2, 3, 100, 100 };
	conf.T = T;
	conf.seed = seed;

// dp: size T, z :size N, w : size N
	int N = ws.size();
	std::vector<int> z(N);
	std::vector<int> dp(D*T);

	gibbs_sampler_lda_predict(conf, &ws[0], &ds[0], N, D, model, &z[0], &dp[0]);

	std::for_each(std::begin(dp), std::end(dp), [&](int p) { fout << p << "\t"; });

	lda_result_destroy(model);

}

#include "cmdline.h"

void configure_parser(cmdline::parser& parser) {

	parser.add<std::string>("action", '\0',
		"action type: preprocess, train, onlinetrain, predict", 
		true, "preprocess", 
		cmdline::oneof<std::string>("preprocess", "train", "onlinetrain", "predict")
	);
}

void print_usage(char* program)
{
	auto usage = fmt::format(
		"usage: {} subcmd [options] \n subcmd:preprocess, train, predict\n", program);
	std::cout<<usage << std::endl;
}

void configure_parser_preprocess(cmdline::parser& parser) {
	parser.add<std::string>("docs", 'd',
		"documents file to be processed",
		true, "" );
	parser.add<std::string>("output", 'o',
		"output file to be saved",
		true, "" );
	parser.add("help", 0, "print this message");
	parser.footer("filename ...");
}

void configure_parser_train(cmdline::parser& parser) {
	parser.add<std::string>("corpus", '\0',
		"corpus file which is the ouput of preprocess subcmd",
		true, "" );
	parser.add<std::string>("output", 'o',
		"output file to be saved",
		true, "" );
	parser.add("help", 0, "print this message");
	parser.footer("filename ...");
}

void configure_parser_predict(cmdline::parser& parser) {
	parser.add<std::string>("model", '\0',
		"model file saved by train subcmd",
		true, "" );
	parser.add<std::string>("docs", 'd',
		"docs file to be predicted",
		true, "" );
	parser.add<std::string>("output", 'o',
		"output file to be saved",
		true, "" );
	parser.add("help", 0, "print this message");
	parser.footer("filename ...");
}

void parser_check(cmdline::parser& parser, int argc, char * * argv)
{
	bool ok = parser.parse(argc, argv);

	if (argc == 1 || parser.exist("help")) {
		std::cerr << parser.usage();
		exit(-1);
	}

	if (!ok) {
		std::cerr << parser.error() << std::endl << parser.usage();
		exit(-1);
	}

}


int main(int argc, char** argv) {

try {
	if (argc == 1) {
		print_usage(argv[0]);
		return -1;
	}
	// create a parser
	cmdline::parser parser;

	std::string subcmd = std::string(argv[1]);
	if (subcmd == "preprocess") {
		parser.set_program_name(std::string(argv[0]) +" " + argv[1]);
		configure_parser_preprocess(parser);
		parser_check(parser, argc - 1, argv + 1);

		auto docs_file = parser.get<std::string>("docs");
		auto out_file = parser.get<std::string>("output");
		process_text3(docs_file, out_file);
	} else if (subcmd == "train") {
		parser.set_program_name(std::string(argv[0]) +" " + argv[1]);
		configure_parser_train(parser);
		parser_check(parser, argc - 1, argv + 1);

		auto corpus_file = parser.get<std::string>("corpus");
		auto out_file = parser.get<std::string>("output");
		int seed = argc;
		lda_train(seed, corpus_file, out_file);
		
	} else if (subcmd == "predict") {
		parser.set_program_name(std::string(argv[0]) +" " + argv[1]);
		configure_parser_predict(parser);
		parser_check(parser, argc - 1, argv + 1);

		auto model_file = parser.get<std::string>("model");
		auto docs_file = parser.get<std::string>("docs");
		auto out_file = parser.get<std::string>("output");
		int seed = argc;
		std::cout << "corpus:" << model_file << "\tdocs:" << docs_file << std::endl;
		lda_predict(seed, model_file, docs_file);

	} else if (subcmd == "wordsimilarity") {
		parser.set_program_name(std::string(argv[0]) +" " + argv[1]);
		configure_parser_predict(parser);
		parser_check(parser, argc - 1, argv + 1);

		auto corpus_file = parser.get<std::string>("model");
		auto docs_file = parser.get<std::string>("docs");
		auto out_file = parser.get<std::string>("output");
		int seed = argc;
		std::cout << "corpus:" << corpus_file << "\tdocs:" << docs_file << std::endl;
	} else {
		print_usage(argv[0]);
		return -1;
	}
	
return		0;

	// boolean flags are referred by calling exist() method.
	if (parser.exist("gzip")) std::cout << "gzip" << std::endl;

}
catch (const cmdline::cmdline_error& e) {
	std::cerr << e.what() << std::endl;
	return -1;
}
catch (const std::exception & e) {
	std::cerr << e.what() << std::endl;
	return -1;
}

	return 0;
}

int __main(int argc, char*argv[])
{
	//process_text();
	lda_words_similary("model-name");
	//heap_test();

	return 0;
}
