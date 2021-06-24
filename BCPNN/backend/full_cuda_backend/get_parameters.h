
//#define BCPNN_FETCH_PARAMETERS

#ifdef BCPNN_FETCH_PARAMETERS
extern "C" {
void get_parameters(char const* s, size_t len, float* taupdt, size_t* l1_epochs,
                    size_t* l2_epochs, float* l1_pmin, float* l1_khalf,
                    float* l1_taubdt);
}
#endif
