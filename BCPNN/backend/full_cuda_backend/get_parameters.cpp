
size_t
my_curl_writefunction(void * buffer, size_t size, size_t nmemb, void *userp)
{
  std::string * s = (std::string *)userp;
  s->append((char *)buffer, size * nmemb);

  return size * nmemb;
}

std::string
make_request(char const* url)
{
  CURL * h = NULL;
  std::string response;

  h = curl_easy_init(); 
  curl_easy_setopt(h, CURLOPT_URL, url);
  curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, my_curl_writefunction);
  curl_easy_setopt(h, CURLOPT_WRITEDATA, &response);
  curl_easy_perform(h);

  return response;
}
