\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{sm-patent}[2026/10/16 Editable technical patent application]
\newif\ifpatentUS
\DeclareOption{us}{\patentUStrue}
\DeclareOption{pct}{\patentUSfalse}
\ProcessOptions\relax
\ifpatentUS\LoadClass[12pt,letterpaper]{article}\else\LoadClass[12pt,a4paper]{article}\fi
\RequirePackage[T1]{fontenc}
\RequirePackage[utf8]{inputenc}
\RequirePackage{mathptmx}
\RequirePackage[top=30mm,bottom=25mm,left=30mm,right=25mm,includehead,headheight=15pt,headsep=8mm]{geometry}
\RequirePackage{setspace,amsmath,enumitem,fancyhdr}
\RequirePackage[pagewise]{lineno}
\RequirePackage[hidelinks,pdfauthor={Tero Keski-Valkama}]{hyperref}
\pagestyle{fancy}\fancyhf{}\fancyhead[C]{\thepage}\renewcommand{\headrulewidth}{0pt}
\setstretch{1.5}\setlength{\parindent}{0pt}\setlength{\parskip}{.6\baselineskip}
\setlength{\emergencystretch}{2em}
\newcounter{patentpara}
\newcommand{\pnum}{\par\refstepcounter{patentpara}\noindent[\ifnum\value{patentpara}<1000 0\fi\ifnum\value{patentpara}<100 0\fi\ifnum\value{patentpara}<10 0\fi\arabic{patentpara}]\enspace}
\newcommand{\PatentTitle}[1]{\gdef\patenttitle{#1}\hypersetup{pdftitle={#1}}}
\newcommand{\pheading}[1]{\par\bigskip\noindent\textbf{#1}\par\nobreak}
\newcommand{\descriptionstart}{\begin{center}\textbf{\MakeUppercase{\patenttitle}}\end{center}\modulolinenumbers[5]\linenumbers}
\newcommand{\claimsstart}{\clearpage\nolinenumbers\begin{center}\textbf{CLAIMS}\end{center}\resetlinenumber[1]\linenumbers}
\newcommand{\abstractstart}{\clearpage\nolinenumbers\begin{center}\textbf{ABSTRACT}\end{center}}
\newenvironment{patentclaims}{\begin{enumerate}[label=\arabic*.,leftmargin=1.8em,itemsep=\baselineskip,parsep=0pt]}{\end{enumerate}}
\endinput
