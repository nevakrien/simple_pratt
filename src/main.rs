use crate::fmt::Debug;
use thiserror::Error;
use std::ops::Deref;
use std::ops::DerefMut;
use std::fmt;

#[derive(Clone,Copy, Debug,PartialEq)]
pub struct Loc {
   pub start:usize,
   pub end:usize,//exclusive
}

impl Loc{
    pub fn merge(self,other:Loc)->Loc{
        Loc{
            start:self.start.min(other.start),
            end:self.end.max(other.end),
        }
    }

    pub fn with<U>(self,value:U)->Located<U>{
        Located{
            value,
            loc:self
        }
    }

    pub fn next_end(self)->Loc{
        Loc{
            start:self.end,
            end:self.end,
        }
    }

    pub fn get_str(self,s:&str) -> &str{
        let start = self.start.min(s.len().saturating_sub(1));
        let end = self.end.min(s.len());
        &s[start..end]
    }
}

#[derive(Clone,PartialEq)]
pub struct Located<T> {
    pub value: T,
    pub loc: Loc,
}

impl<T> Located<T> {
    #[inline]
    pub fn new(value: T, loc: Loc) -> Self {
        Self { value, loc }
    }


    pub fn into_inner(self)->T{
        self.value
    }

    pub fn map_owned<U>(self,f:impl Fn(T)->U)->Located<U>{
        Located{
            value:f(self.value),
            loc:self.loc
        }
    }

    pub fn with<U>(&self,value:U)->Located<U>{
        Located{
            value,
            loc:self.loc.clone()
        }
    }

    pub fn fixtype<U:From<T>>(self)->Located<U>{
        Located{
            value: self.value.into(),
            loc:self.loc
        }
    }
}

impl<T: fmt::Display> fmt::Display for Located<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.value.fmt(f)
    }
}

impl <T:Debug> Debug for Located<T>{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f,"Located({:?})",self.value)
    }

}

impl<T> Deref for Located<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.value
    }
}

impl<T> DerefMut for Located<T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum Operator {
    // cast
    As,

    // Arithmetic
    Plus,      // +
    Minus,     // -
    Star,      // *
    Slash,     // /
    Percent,   // %

    // Bitwise
    Amp,       // &
    Pipe,      // |
    Caret,     // ^
    Tilde,     // ~
    Shl,       // <<
    Shr,       // >>

    // Logical / comparison
    Bang,      // !
    EqEq,      // ==
    Ne,        // !=
    Lt,        // <
    Le,        // <=
    Gt,        // >
    Ge,        // >=
    LAnd,      // &&
    LOr,       // ||

    // Assignment / pointer
    Assign,    // =
    Arrow,     // ->
    // Amp and Star cover & (address) and * (deref)

    // Grouping / delimiters
    LParen, RParen,
    LBrace, RBrace,
    LBracket, RBracket,

    Comma, Semicolon,Dot,
}

impl Operator{
    pub fn as_str(self)->&'static str{
        use Operator::*;
        match self {
            // cast
            As => "as",

            // Arithmetic
            Plus => "+",
            Minus => "-",
            Star => "*",
            Slash => "/",
            Percent => "%",

            // Bitwise
            Amp => "&",
            Pipe => "|",
            Caret => "^",
            Tilde => "~",
            Shl => "<<",
            Shr => ">>",

            // Logical / comparison
            Bang => "!",
            EqEq => "==",
            Ne   => "!=",
            Lt   => "<",
            Le   => "<=",
            Gt   => ">",
            Ge   => ">=",
            LAnd => "&&",
            LOr  => "||",

            // Assignment / pointer
            Assign => "=",
            Arrow  => "->",

            // Grouping / delimiters
            LParen   => "(",
            RParen   => ")",
            LBrace   => "{",
            RBrace   => "}",
            LBracket => "[",
            RBracket => "]",

            Comma => ",",
            Semicolon => ";",
            Dot => ".",
        }
    }
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, PartialEq, Hash)]
pub enum Token<'a>{
    Op(Operator),
    Name(&'a str),
    Num(u64),
    Str(String),
}

#[derive(Debug, Error,Clone,PartialEq,Eq)]
pub enum LexError {
    #[error("invalid number ending with '{0}'")]
    WeirdNumberEnd(char),
    // UnKnowenEscape(char),

    #[error("missing closing quote in string")]
    MissingStringClose,

    #[error("unknown token '{0}'")]
    UnknownChar(char),
}



type LexRes<T> = Result<T,Located<LexError>>;

#[derive(Debug)]
pub struct Lexer<'a>{
    // original_str:&'a str,

    peeked:Option<Located<Token<'a>>>,
    cur_str:&'a str,
    cur_start:usize,
}


impl<'a> Lexer<'a>{
    pub fn new(original_str:&'a str)->Self{
        Self{
            // original_str,
            peeked:None,
            cur_str:original_str,
            cur_start:0,
        }
    }

    pub fn peek(&mut self)->LexRes<Option<&Located<Token<'a>>>>{
        if self.peeked.is_none(){
            self.peeked=self.next()?;
        }
        Ok(self.peeked.as_ref())

    }

    fn skip_whitespace(&mut self){
        for c in self.cur_str.chars(){
            if !c.is_whitespace(){
                return;
            }

            let size = c.len_utf8();
            self.cur_start+=size;
            self.cur_str=&self.cur_str[size..];
        }
    }

    fn skip_comments(&mut self){
        loop{
            self.skip_whitespace();
            if self.cur_str.as_bytes().get(..2) != Some(b"//"){
                return;
            }
            self.yeild_next(2);
            for c in self.cur_str.chars(){
                if c=='\n'{
                    break;
                }

                let size = c.len_utf8();
                self.cur_start+=size;
                self.cur_str=&self.cur_str[size..];
            }
        }
    }

    fn yeild_next(&mut self,size:usize)->Located<&'a str>{
        let value = &self.cur_str[..size];
        let end = self.cur_start+size;

        let loc = Loc{
            start:self.cur_start,
            end,
        };

        let ans = Located::new(value,loc);

        if end-self.cur_start < self.cur_str.len(){
            self.cur_str=&self.cur_str[size..];
        }else{
            self.cur_str="";
        }
        self.cur_start=end;
        ans
    }

    fn try_parse_raw(&mut self,s:&str)->Option<Located<&'a str>>{
        let found = self.cur_str.get(..s.len())?;
        if found==s{
            Some(self.yeild_next(s.len()))
        }else{
            None
        }
    }

    fn parse_name(&mut self)->Located<Token<'a>>{
        let mut size = 0usize;
        for c in self.cur_str.chars(){
            if !(c.is_alphanumeric() || c=='_') {
                break;
            }

            size+=c.len_utf8();
        }

        self.yeild_next(size).map_owned(Token::Name)
    }

    fn parse_number(&mut self)->LexRes<Located<Token<'a>>>{
        let mut size = 0usize;
        for c in self.cur_str.chars(){
            if c.is_whitespace() {
                break;
            }

            if !(c.is_numeric() || c=='_') {
                if !c.is_alphabetic(){
                    break;//we explictly allow 2+3 style parses
                }
                size+=c.len_utf8();
                let tok = self.yeild_next(size);
                let err = tok.map_owned(|_|{LexError::WeirdNumberEnd(c)});
                return Err(err);
            }

            size+=c.len_utf8();
        }

        let tok = self.yeild_next(size);
        let mut num = 0u64;
        for c in tok.as_bytes().iter() {
            if *c==b'_'{
                continue;
            }

            num=10*num+((c-b'0') as u64);

        }
        Ok(tok.with(Token::Num(num)))
    }

    fn parse_string(&mut self)->LexRes<Located<Token<'a>>>{
        let mut size = 1;

        let mut s = String::new();
        let mut skip = false;

        for c in self.cur_str[1..].chars(){
            size+=c.len_utf8();
            
            if c == '\n'{
                break
            }

            if skip {
                skip = false;
                let resolved = match c {
                    'n'  => '\n',
                    'r'  => '\r',
                    't'  => '\t',
                    '\\' => '\\',
                    '"'  => '"',
                    '\'' => '\'',
                    '0'  => '\0',
                    // 'v'  => '\x0B',
                    // 'f'  => '\x0C',

                    //there is a good argument for erroring here
                    //the issue is we dont wana compltly destroy the parse...
                    //which implies multi error and thats really tricky because heap...
                    //so for now we dont bother
                    _    => c,//return Err(self.yeild_next(size).with(LexError::UnKnowenEscape(c))),
                };

                s.push(resolved);
                continue;
            }


            if c=='"' {
                return Ok(self.yeild_next(size).with(Token::Str(s)));
            }

            if c=='\\' {
                skip = true;
                continue;
            }

            s.push(c);

        }

        Err(self.yeild_next(size).with(LexError::MissingStringClose))
    }

    fn parse_operator(&mut self) -> Option<Located<Token<'a>>> {
        macro_rules! op {
            ($tok:expr, $op:expr) => {
                return Some($tok.map_owned(|_| Token::Op($op)))
            };
        }

        // multi-character first
        if let Some(tok) = self.try_parse_raw("as") { op!(tok, Operator::As); }
        if let Some(tok) = self.try_parse_raw("==") { op!(tok, Operator::EqEq); }
        if let Some(tok) = self.try_parse_raw("!=") { op!(tok, Operator::Ne); }
        if let Some(tok) = self.try_parse_raw("<=") { op!(tok, Operator::Le); }
        if let Some(tok) = self.try_parse_raw(">=") { op!(tok, Operator::Ge); }
        if let Some(tok) = self.try_parse_raw("&&") { op!(tok, Operator::LAnd); }
        if let Some(tok) = self.try_parse_raw("||") { op!(tok, Operator::LOr); }
        if let Some(tok) = self.try_parse_raw("<<") { op!(tok, Operator::Shl); }
        if let Some(tok) = self.try_parse_raw(">>") { op!(tok, Operator::Shr); }
        if let Some(tok) = self.try_parse_raw("->") { op!(tok, Operator::Arrow); }

        // single-character symbols
        if let Some(tok) = self.try_parse_raw("+") { op!(tok, Operator::Plus); }
        if let Some(tok) = self.try_parse_raw("-") { op!(tok, Operator::Minus); }
        if let Some(tok) = self.try_parse_raw("*") { op!(tok, Operator::Star); }
        if let Some(tok) = self.try_parse_raw("/") { op!(tok, Operator::Slash); }
        if let Some(tok) = self.try_parse_raw("%") { op!(tok, Operator::Percent); }

        if let Some(tok) = self.try_parse_raw("&") { op!(tok, Operator::Amp); }
        if let Some(tok) = self.try_parse_raw("|") { op!(tok, Operator::Pipe); }
        if let Some(tok) = self.try_parse_raw("^") { op!(tok, Operator::Caret); }
        if let Some(tok) = self.try_parse_raw("~") { op!(tok, Operator::Tilde); }

        if let Some(tok) = self.try_parse_raw("<") { op!(tok, Operator::Lt); }
        if let Some(tok) = self.try_parse_raw(">") { op!(tok, Operator::Gt); }
        if let Some(tok) = self.try_parse_raw("=") { op!(tok, Operator::Assign); }
        if let Some(tok) = self.try_parse_raw("!") { op!(tok, Operator::Bang); }

        // delimiters / punctuation
        if let Some(tok) = self.try_parse_raw("(") { op!(tok, Operator::LParen); }
        if let Some(tok) = self.try_parse_raw(")") { op!(tok, Operator::RParen); }
        if let Some(tok) = self.try_parse_raw("{") { op!(tok, Operator::LBrace); }
        if let Some(tok) = self.try_parse_raw("}") { op!(tok, Operator::RBrace); }
        if let Some(tok) = self.try_parse_raw("[") { op!(tok, Operator::LBracket); }
        if let Some(tok) = self.try_parse_raw("]") { op!(tok, Operator::RBracket); }
        if let Some(tok) = self.try_parse_raw(",") { op!(tok, Operator::Comma); }
        if let Some(tok) = self.try_parse_raw(";") { op!(tok, Operator::Semicolon); }
        if let Some(tok) = self.try_parse_raw(".") { op!(tok, Operator::Dot); }

        None
    }


    pub fn next(&mut self)->LexRes<Option<Located<Token<'a>>>>{
        if self.peeked.is_some(){
            return Ok(self.peeked.take())
        }
        self.skip_comments();
        let Some(c) = self.cur_str.chars().next() else{
            return Ok(None);
        };
        
        if c == '"'{
            return Ok(Some(self.parse_string()?))   
        }

        if let Some(ans) = self.parse_operator(){
            return Ok(Some(ans))
        }

        if c.is_alphabetic() || c=='_' {
            return Ok(Some(self.parse_name()));
        }
        if c.is_numeric(){
            return Ok(Some(self.parse_number()?));
        }

        let loc = self.yeild_next(c.len_utf8());
        Err(loc.with(LexError::UnknownChar(c)))
    }

    #[inline(always)]
    pub fn tuck(&mut self,tok:Located<Token<'a>>){
        assert!(self.peeked.is_none());
        self.peeked=Some(tok)
    }

    // fn str_of(&self,loc:Loc)->&'a str{
    //     &self.original_str[loc.start..loc.end]
    // }

}


#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnOp {
    Neg, Pos, Not, BitNot, Addr, Deref,
    PostCall, PostIndex, PostCast,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add, Sub, Mul, Div, Mod,
    Shl, Shr,
    Lt, Le, Gt, Ge, Eq, Ne,
    BitAnd, BitOr, BitXor,
    And, Or,
    Assign,
    Member, PtrMember, Index,
}

pub type LocType = Located<()>;

pub type LocExpr<'a> = Located<Expr<'a>>;

#[derive(Debug)]
pub enum Expr<'a> {
    Var(&'a str),
    IntLit(u64),
    StrLit(String),
    UnOp(UnOp,Box<LocExpr<'a>>),
    BinOp(BinOp,Box<[LocExpr<'a>;2]>),
    Call(Box<LocExpr<'a>>,Box<[LocExpr<'a>]>),
    Cast(Box<LocExpr<'a>>,LocType)
}

pub type Bp = u32;

impl Operator {
    pub fn get_prefix(self) -> Option<(UnOp, Bp)> {
        Some(match self {
            Operator::Plus   => (UnOp::Pos,    80),
            Operator::Minus  => (UnOp::Neg,    80),
            Operator::Bang   => (UnOp::Not,    80),
            Operator::Tilde  => (UnOp::BitNot, 80),
            Operator::Star   => (UnOp::Deref,  80),
            Operator::Amp    => (UnOp::Addr,   80),
            _ => return None,
        })
    }

    pub fn get_postfix(self) -> Option<(Bp, UnOp)> {
        Some(match self {
            Operator::LParen   => (90, UnOp::PostCall),
            Operator::LBracket => (90, UnOp::PostIndex),
            Operator::As       => (90, UnOp::PostCast),
            _ => return None,
        })
    }

    pub fn get_infix(self) -> Option<(Bp, BinOp, Bp)> {
        Some(match self {
            // Member access are binary postfix
            Operator::Dot      => (90, BinOp::Member, 91),
            Operator::Arrow    => (90, BinOp::PtrMember, 91),

            // Multiplicative
            Operator::Star     => (70, BinOp::Mul, 71),
            Operator::Slash    => (70, BinOp::Div, 71),
            Operator::Percent  => (70, BinOp::Mod, 71),

            // Additive
            Operator::Plus     => (60, BinOp::Add, 61),
            Operator::Minus    => (60, BinOp::Sub, 61),

            // Shifts
            Operator::Shl      => (55, BinOp::Shl, 56),
            Operator::Shr      => (55, BinOp::Shr, 56),

            // Comparisons
            Operator::Lt       => (50, BinOp::Lt, 51),
            Operator::Le       => (50, BinOp::Le, 51),
            Operator::Gt       => (50, BinOp::Gt, 51),
            Operator::Ge       => (50, BinOp::Ge, 51),

            // Equalities
            Operator::EqEq     => (45, BinOp::Eq, 46),
            Operator::Ne       => (45, BinOp::Ne, 46),

            // Bitwise
            Operator::Amp      => (40, BinOp::BitAnd, 41),
            Operator::Caret    => (35, BinOp::BitXor, 36),
            Operator::Pipe     => (30, BinOp::BitOr, 31),

            // Logical
            Operator::LAnd     => (25, BinOp::And, 26),
            Operator::LOr      => (20, BinOp::Or,  21),

            // Assignment
            Operator::Assign   => (10, BinOp::Assign, 9),

            _ => return None,
        })
    }
}



pub type ParseRes<'a, T> = Result<T, Located<ParseError<'a>>>;

#[derive(Debug, Clone, Error, PartialEq)]
pub enum ParseError<'a> {
    #[error("lex error: {0}")]
    Lex(#[from] LexError),

    #[error("\"{0}\" is not a recognized postfix or infix operator")]
    NotPost(Operator),

    #[error("\"{0}\" is not a recognized prefix operator")]
    NotPre(Operator),

    #[error("\"{0}\" is not a recognized postfix or infix operator")]
    NeedRight(Located<&'a str>),

    #[error("expected {0} found \"{1}\"")]
    Expected(&'static str,&'a str),

    #[error("expected {0} found EOF")]
    EarlyEOF(&'static str),
}

impl<'a> From<Located<LexError>> for Located<ParseError<'a>>{
fn from(x: Located<LexError>) -> Self { x.fixtype() }
}

pub struct Parser<'a>{
    end_span:Loc,
    original_str:&'a str,
    lexer:Lexer<'a>,
}

impl<'a> Parser<'a>{
    pub fn new(original_str:&'a str)->Self{
        let end_span = Loc {
            start:original_str.len().saturating_sub(1),
            end:original_str.len(),
        };
        Self{
            end_span,
            original_str,
            lexer:Lexer::new(original_str)
        }
    }
    pub fn try_consume(&mut self,want:&str)->ParseRes<'a,Option<Located<&'a str>>>{
        let Some(top) =  self.lexer.peek()? else {
            return Ok(None)
        };

        let loc = top.loc;
        let found = loc.get_str(self.original_str);
        if found!=want{
            return Ok(None)
        }

        self.lexer.next()?;
        Ok(Some(loc.with(found)))
    }

    pub fn consume(&mut self,want:&'static str)->ParseRes<'a,Loc>{
        let Some(top) =  self.lexer.peek()? else {
            let error = ParseError::EarlyEOF(want);
            return Err(self.end_span.with(error));
        };

        let loc = top.loc;
        let found = loc.get_str(self.original_str);
        if found!=want{
            let error = ParseError::Expected(want,found);
            return Err(loc.with(error));
        }

        self.lexer.next()?;
        Ok(loc)
    }


    pub fn expr_bp(&mut self, min_bp: Bp) -> ParseRes<'a,LocExpr<'a>> {
        
        let Some(tok) = self.lexer.next()? else {
            return Err(self.end_span.with(ParseError::EarlyEOF("value")))
        };

        // ─────────────────────────────
        // atom/pretfix Pratt recursion
        // ─────────────────────────────

        let mut lhs = match tok.value {
            Token::Num(x) => tok.with(Expr::IntLit(x)),
            Token::Name(x) => tok.with(Expr::Var(x)),
            Token::Str(x) => tok.loc.with(Expr::StrLit(x)),
            Token::Op(Operator::LParen) => {
                let lhs= self.expr_bp( 0)?;
                self.consume(")")?;
                lhs
            },
            Token::Op(op) => {
                let Some((op, r_bp)) = op.get_prefix() else{
                    return Err(tok.with(ParseError::NotPre(op)))
                };
                let rhs = self.expr_bp(r_bp)?;
                let loc = tok.loc.merge(rhs.loc);
                loc.with(Expr::UnOp(op, Box::new(rhs)))
            }
        };

        // ─────────────────────────────
        // infix/postfix Pratt loop
        // ─────────────────────────────
        loop {
            // Peek next token
            let Some(peek) = self.lexer.peek()? else { break };
            let op_loc = peek.loc;


            let op :Operator = match &peek.value {
                Token::Op(op) => *op,
                _ => {
                    break;
                },
            };

            // ───────────── INFIX ─────────────
            if let Some((l_bp, bin_op, r_bp)) = op.get_infix() {
                if l_bp < min_bp {
                    break;
                }

                self.lexer.next()?;
                let rhs = self.expr_bp(r_bp)?;
                let span = lhs.loc.merge(rhs.loc);
                lhs = span.with(Expr::BinOp(bin_op, Box::new([lhs, rhs])));
                continue;
            }

            // ───────────── POSTFIX ─────────────
            if let Some((l_bp, post_op)) = op.get_postfix() {
                if l_bp < min_bp { break; }

                // consume the operator token
                self.lexer.next()?;

                match post_op {
                    // f(args…)
                    UnOp::PostCall => {
                        let mut args = Vec::new();
                        if self.try_consume(")")?.is_none() {
                            loop {
                                let arg = self.expr_bp(0)?;
                                args.push(arg);
                                if self.try_consume(",")?.is_some() { continue; }
                                break;
                            }
                            self.consume(")")?;
                        }
                        let end = args.last().map(|a| a.loc.end).unwrap_or(lhs.loc.end);
                        let span = Loc { start: lhs.loc.start, end };
                        lhs = span.with(Expr::Call(Box::new(lhs), args.into_boxed_slice()));
                    }

                    // a[expr]
                    UnOp::PostIndex => {
                        let rhs = self.expr_bp(0)?;
                        self.consume("]")?;
                        let span = Loc { start: lhs.loc.start, end: rhs.loc.end };
                        lhs = span.with(Expr::BinOp(BinOp::Index, Box::new([lhs, rhs])));
                    }

                    // a as Type
                    UnOp::PostCast => {
                        // integrate your type parser here later
                        todo!("type parser for cast expression");
                    }

                    // default: postfix op with no extra tokens
                    _ => {
                        let span = Loc { start: lhs.loc.start, end: op_loc.end };
                        lhs = span.with(Expr::UnOp(post_op, Box::new(lhs)));
                    }
                }

                continue;
            }


            

            // stop if not postfix/infix
            break;
        }


        Ok(lhs)
    }
    
}

use ariadne::{Color, Label, Report, ReportKind, Source};
use std::ops::Range;

/// Pretty-print a single parse error using Ariadne.
pub fn report_parse_error<'a>(
    file_name: &str,
    src: &'a str,
    err: &Located<ParseError<'a>>,
) {
    let range: Range<usize> = err.loc.start..err.loc.end;
    let msg = err.value.to_string();

    Report::build(ReportKind::Error, (file_name, range.clone()))
        .with_message("parse error")
        .with_label(
            Label::new((file_name, range))
                .with_message(msg)
                .with_color(Color::Red),
        )
        .finish()
        .print((file_name, Source::from(src)))
        .unwrap();
}

use std::io::{self, Write};

fn main() {
    println!("🦀 simple_pratt REPL — enter expressions, Ctrl+D to exit");

    let stdin = io::stdin();
    let mut input = String::new();

    loop {
        print!("> ");
        io::stdout().flush().unwrap();
        input.clear();

        if stdin.read_line(&mut input).unwrap() == 0 {
            break;
        }

        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }

        let mut parser = Parser::new(trimmed);

        match parser.expr_bp(0) {
            Ok(expr) => {
                println!("✅ Parsed successfully: {:#?}", expr.value);
            }
            Err(err) => {
                println!("❌ Parse error:\n");
                report_parse_error("<repl>", trimmed, &err);
            }
        }
    }

    println!("bye 👋");
}
