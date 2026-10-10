#include <torch/extension.h>
#include <vector>

static void check(torch::Tensor x) {
  TORCH_CHECK(x.device().is_cpu() && x.scalar_type()==torch::kFloat64 && x.is_contiguous(),
              "CPU scan requires contiguous float64 tensors");
}

torch::Tensor scan_forward(torch::Tensor cr, torch::Tensor ci, torch::Tensor wr,
                           torch::Tensor wi, torch::Tensor initial) {
  check(cr);check(ci);check(wr);check(wi);check(initial);
  TORCH_CHECK(cr.dim()==3 && ci.sizes()==cr.sizes() && wr.sizes()==cr.sizes() && wi.sizes()==cr.sizes(), "coefficient shape");
  auto B=cr.size(0),L=cr.size(1),N=cr.size(2);
  TORCH_CHECK(L>0 && initial.dim()==2 && initial.size(0)==B && initial.size(1)==2*N,"initial shape");
  auto out=torch::empty({B,L,2*N},cr.options());
  const auto *ar=cr.data_ptr<double>(),*ai=ci.data_ptr<double>(),*ur=wr.data_ptr<double>(),*ui=wi.data_ptr<double>(),*s=initial.data_ptr<double>();
  auto *z=out.data_ptr<double>();
  for(int64_t b=0;b<B;b++) for(int64_t t=0;t<L;t++) for(int64_t n=0;n<N;n++) {
    auto k=(b*L+t)*N+n, o=(b*L+t)*2*N+n;
    double pr=t?z[o-2*N]:s[b*2*N+n], pi=t?z[o-2*N+N]:s[b*2*N+N+n];
    z[o]=(ar[k]*pr-ai[k]*pi)+ur[k];
    z[o+N]=(ai[k]*pr+ar[k]*pi)+ui[k];
  }
  return out;
}

std::vector<torch::Tensor> scan_backward(torch::Tensor cr, torch::Tensor ci,
    torch::Tensor initial, torch::Tensor out, torch::Tensor upstream) {
  check(cr);check(ci);check(initial);check(out);check(upstream);
  auto B=cr.size(0),L=cr.size(1),N=cr.size(2);
  TORCH_CHECK(out.dim()==3 && out.size(0)==B && out.size(1)==L && out.size(2)==2*N && upstream.sizes()==out.sizes(),"gradient shape");
  auto gar=torch::empty_like(cr), gai=torch::empty_like(ci), gur=torch::empty_like(cr), gui=torch::empty_like(ci), gs=torch::empty_like(initial);
  const auto *ar=cr.data_ptr<double>(),*ai=ci.data_ptr<double>(),*s=initial.data_ptr<double>(),*z=out.data_ptr<double>(),*g=upstream.data_ptr<double>();
  auto *dr=gar.data_ptr<double>(),*di=gai.data_ptr<double>(),*du=gur.data_ptr<double>(),*dv=gui.data_ptr<double>(),*ds=gs.data_ptr<double>();
  for(int64_t b=0;b<B;b++) for(int64_t n=0;n<N;n++) {
    double br=0.,bi=0.;
    for(int64_t t=L;t-->0;) {
      auto k=(b*L+t)*N+n,o=(b*L+t)*2*N+n;
      br+=g[o];bi+=g[o+N];
      double pr=t?z[o-2*N]:s[b*2*N+n], pi=t?z[o-2*N+N]:s[b*2*N+N+n];
      du[k]=br;dv[k]=bi;dr[k]=br*pr+bi*pi;di[k]=-br*pi+bi*pr;
      double prev_r=ar[k]*br+ai[k]*bi,prev_i=-ai[k]*br+ar[k]*bi;
      br=prev_r;bi=prev_i;
    }
    ds[b*2*N+n]=br;ds[b*2*N+N+n]=bi;
  }
  return {gar,gai,gur,gui,gs};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("forward",&scan_forward);m.def("backward",&scan_backward);
}
