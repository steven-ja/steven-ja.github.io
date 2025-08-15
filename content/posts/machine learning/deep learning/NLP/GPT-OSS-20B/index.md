# GPT-OSS 20B - Tutorial
## 1. Install Dependencies and Imports

    📦 Installing required packages...
    ✅ Packages installed successfully!
    

## 2. Setup Display Functions

## 3. Install and Start Ollama



<div style="padding: 10px; margin: 10px 0; border-left: 4px solid #9b59b6; background-color: #f8f9fa;">
    <strong style="color: #9b59b6;">🚀 Setting up Ollama...</strong>
</div>



    Installing Ollama... This may take a minute...



<div style="padding: 10px; margin: 10px 0; border-left: 4px solid #2ecc71; background-color: #f8f9fa;">
    <strong style="color: #2ecc71;">✅ Ollama installed successfully!</strong>
</div>



    Starting Ollama server...
    



<div style="padding: 10px; margin: 10px 0; border-left: 4px solid #2ecc71; background-color: #f8f9fa;">
    <strong style="color: #2ecc71;">✅ Ollama server is running!</strong>
</div>



## 4. Download Model with Progress



<div style="padding: 10px; margin: 10px 0; border-left: 4px solid #9b59b6; background-color: #f8f9fa;">
    <strong style="color: #9b59b6;">📥 Downloading GPT-OSS:20B Model (~13GB)</strong>
</div>



    This will take several minutes. Please be patient...
    ============================================================
    



<div style="padding: 10px; margin: 10px 0; border-left: 4px solid #2ecc71; background-color: #f8f9fa;">
    <strong style="color: #2ecc71;">✅ Model downloaded successfully in 1.4 minutes!</strong>
</div>



    
    📋 Available models:
    NAME           ID              SIZE     MODIFIED               
    gpt-oss:20b    e95023cf3b7b    13 GB    Less than a second ago    
    



## 5. Setup Model Interface



<div style="padding: 10px; margin: 10px 0; border-left: 4px solid #9b59b6; background-color: #f8f9fa;">
    <strong style="color: #9b59b6;">🎉 Initializing GPT-OSS Chat Interface...</strong>
</div>





<div style="padding: 10px; margin: 10px 0; border-left: 4px solid #2ecc71; background-color: #f8f9fa;">
    <strong style="color: #2ecc71;">✅ Chat interface ready!</strong>
</div>



## 6. Test the Model



<div style="background-color: #ecf0f1; padding: 20px; border-radius: 10px; margin: 20px 0;">
    <h2>7. 📚 Usage Examples</h2>
    <p>Here are different ways to use the model:</p>
</div>



### Example 1

 ### Examples 2-3-4


**You:** Write a Python function to calculate fibonacci numbers



**Assistant:** Below is a tiny, self‑contained Python module that gives you a handful of ways to ask for Fibonacci numbers.  
Feel free to pick the one that best matches your use‑case (simple recursion, memoised recursion, or an iterative loop that is fast even for millions o terms).

```python
from __future__ import annotations
from typing import Iterable, List, Sequence

# ------------------------------------------------------------------
# 1️⃣  Simple pure recursion (good for tiny values of n)
# ------------------------------------------------------------------
def fib_recursive(n: int) -> int:
    """Return the *n*‑th Fibonacci number (0‑based).

    >>> fib_recursive(0)
    0
    >>> fib_recursive(1)
    1
    >>> fib_recursive(10)
    55
    """
    if n < 0:
        raise ValueError("n must be a non‑negative integer.")
    if n <= 1:
        return n
    return fib_recursive(n - 1) + fib_recursive(n - 2)


# ------------------------------------------------------------------
# 2️⃣  Recursive + memoisation (fast for repeated calls)
# ------------------------------------------------------------------
def fib_memoised(n: int, cache: dict[int, int] | None = None) -> int:
    """Return the *n*‑th Fibonacci number with memoisation.

    The optional *cache* dict is used internally – you can supply one to
    share memoisation across multiple calls.

    >>> fib_memoised(50)                # 12586269025
    12586269025
    >>> cache: dict[int, int] = {}
    >>> fib_memoised(50, cache=cache)  # uses cache
    12586269025
    """
    if cache is None:
        cache = {}
    if n < 0:
        raise ValueError("n must be a non‑negative integer.")
    if n in cache:
        return cache[n]
    if n <= 1:
        cache[n] = n
    else:
        cache[n] = fib_memoised(n - 1, cache) + fib_memoised(n - 2, cache)
    return cache[n]


# ------------------------------------------------------------------
# 3️⃣  Iterative, constant‑space (fastest for large n)
# ------------------------------------------------------------------
def fib_iterative(n: int) -> int:
    """Return the *n*‑th Fibonacci number using an O(1) space loop.

    >>> fib_iterative(10)
    55
    >>> fib_iterative(100)
    354224848179261915075
    """
    if n < 0:
        raise ValueError("n must be a non‑negative integer.")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


# ------------------------------------------------------------------
# 4️⃣  Generator – yields as many Fibonacci numbers as you need
# ------------------------------------------------------------------
def fibonacci_generator() -> Iterable[int]:
    """Yield Fibonacci numbers forever (0, 1, 1, 2, …)."""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b


# ------------------------------------------------------------------
# 5️⃣  Build a list of the first *n* Fibonacci numbers
# ------------------------------------------------------------------
def first_n_fibonacci(n: int) -> Sequence[int]:
    """Return a list of the first *n* Fibonacci numbers (0‑based).

    >>> first_n_fibonacci(7)
    [0, 1, 1, 2, 3, 5, 8]
    """
    if n < 0:
        raise ValueError("n must be a non‑negative integer.")
    seq: List[int] = []
    for i in range(n):
        seq.append(fib_iterative(i))
    return seq


# ------------------------------------------------------------------
# Quick sanity tests (run `python -m pytest` if you set this up as a test file)
# ------------------------------------------------------------------
if __name__ == "__main__":
    for func in [fib_recursive, fib_memoised, fib_iterative]:
        for n in [0, 1, 5, 10]:
            assert func(n) == fib_iterative(n), f"{func.__name__} failed on n={n}"
    print("All basic checks passed!")
```

### When should you use each one?

| Function | Speed | Memory | Use‑case |
|----------|-------|--------|----------|
| `fib_recursive` | O(2ⁿ) – terrible for `n>30` | Very little | Educational or trivial `n` |
| `fib_memoised` | O(n) | O(n) | Many successive calls where `n` varies |
| `fib_iterative` | O(n) | O(1) | One big Fibonacci number |
| `fibonacci_generator` | O(1) per value | O(1) | Streaming Fibonacci sequence (e.g. plotting) |
| `first_n_fibonacci` | O(n log φ) via iterative per element | O(n) | Build a list of the first few values |

Pick the variant that matches your performance & memory constraints!




<div style="padding: 10px; margin: 10px 0; border-left: 4px solid #3498db; background-color: #f8f9fa;">
    <strong style="color: #3498db;">System message updated: You are a helpful AI assistant....</strong>
</div>





<div style="padding: 10px; margin: 10px 0; border-left: 4px solid #3498db; background-color: #f8f9fa;">
    <strong style="color: #3498db;">Conversation history cleared</strong>
</div>



## 8. Interactive Chat Session

## 9. Quick Chat Function

## 10. Trobleshooting Utilities



<div style="background-color: #f39c12; color: white; padding: 20px; border-radius: 10px; margin: 20px 0;">
    <h2>🔧 Troubleshooting Utilities</h2>
    <p>Use these functions if you encounter any issues</p>
</div>





<div style="padding: 10px; margin: 10px 0; border-left: 4px solid #3498db; background-color: #f8f9fa;">
    <strong style="color: #3498db;">Checking Ollama status...</strong>
</div>





<div style="padding: 10px; margin: 10px 0; border-left: 4px solid #2ecc71; background-color: #f8f9fa;">
    <strong style="color: #2ecc71;">✅ Ollama server is running</strong>
</div>





<div style="padding: 10px; margin: 10px 0; border-left: 4px solid #2ecc71; background-color: #f8f9fa;">
    <strong style="color: #2ecc71;">✅ Ollama API is responding</strong>
</div>



    
    Available models:
    NAME           ID              SIZE     MODIFIED      
    gpt-oss:20b    e95023cf3b7b    13 GB    4 minutes ago    
    

## 11. Custom Chat Widgets


## 12. Save and Load Conversations



<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 15px; margin: 20px 0; text-align: center;">
    <h1>🎉 GPT-OSS:20B is Ready!</h1>
    <p style="font-size: 18px;">Your model is set up and ready to use.</p>
    <p>Use <code>chat.chat("your message")</code> to interact with the model</p>
    <p>Or uncomment <code>chat.interactive_chat()</code> for an interactive session</p>
</div>




**You:** What is the definition of illegal recruitment according to Philippines law?



**Assistant:** **Illegal recruitment** – as defined in the Philippine labor law – is the *unsanctioned procurement and placement of prospective workers* that violates the licensing rules of the Department of Labor and Employment (DLE).  
The definition is codified in the Labor Code of the Philippines (Republic Act No. 7725) under **Section 297** (and reiterated in Section 297‑b). The key points are:

| Clause | What it Covers | Practical Implication |
|--------|----------------|-----------------------|
| **297‑a** | *Solicitation and procurement of prospective employees for employment or placement **without a license** issued by the DLE.* | Any individual or firm that advertises or “recruits” workers for a job, yet has not been officially licensed as a recruitment agency, is acting illegally. |
| **297‑a (continued)** | *Procurement or placement by a licensed agency that **fails to comply** with the terms or conditions of its license.* | Even a licensed recruiter who breaches its license terms (e.g., hiring workers outside its permitted scope) is guilty of illegal recruitment. |
| **297‑a (continued)** | *Procurement or placement by an employee of a licensed agency who **lacks a written directive** from the agency’s master license to place a prospective employee.* | A recruiter inside a licensed agency who works without the agency’s explicit permission is also unlawful. |
| **297‑b** | *Any person who procures a prospective worker for a position **in exchange for a fee or other incentive** (monetary or otherwise), or who places a worker under false or deceptive pretenses (misrepresenting the nature of the job, contract, or benefits).* | Charging recruiters or agents to secure a job for someone, or lying about job terms, is prohibited. |
| **297‑c** | *The placement of a prospective employee in a job that is **inconsistent with the workers’ actual qualifications** or the employer’s legitimate needs.* | “Exploiting a worker’s situation (e.g., hiring an unqualified person when a qualified one is available) for personal gain” is illegal. |

---

### Why It Matters

- **Licensing protects workers** from exploitation, ensures fairness, and guarantees that employers meet labor standards.
- The DLE enforces this definition through inspections, permits, and penalties (see Section 299 for fines and imprisonment).
- Illegal recruitment is a **criminal offense** and can lead to fines, suspension, or jail time for the recruiter or hiring party.

---

#### Quick Reference

- **Law**: Labor Code of the Philippines, **Section 297** (illegal recruitment).
- **Authority**: Department of Labor and Employment (DLE) – Office of the Chief Labor Officer.
- **Penalty**: Section 299 – up to a fine of PHP 35,000, imprisonment of up to 3 years, or both.

By keeping recruitment practices within the limits of this definition, employers, recruiters, and workers can avoid legal trouble and ensure fair, transparent employment arrangements.





    '**Illegal recruitment** – as defined in the Philippine labor law – is the *unsanctioned procurement and placement of prospective workers* that violates the licensing rules of the Department of Labor and Employment (DLE).  \nThe definition is codified in the Labor Code of the Philippines (Republic Act No.\u202f7725) under **Section\u202f297** (and reiterated in Section\u202f297‑b). The key points are:\n\n| Clause | What it Covers | Practical Implication |\n|--------|----------------|-----------------------|\n| **297‑a** | *Solicitation and procurement of prospective employees for employment or placement **without a license** issued by the DLE.* | Any individual or firm that advertises or “recruits” workers for a job, yet has not been officially licensed as a recruitment agency, is acting illegally. |\n| **297‑a (continued)** | *Procurement or placement by a licensed agency that **fails to comply** with the terms or conditions of its license.* | Even a licensed recruiter who breaches its license terms (e.g., hiring workers outside its permitted scope) is guilty of illegal recruitment. |\n| **297‑a (continued)** | *Procurement or placement by an employee of a licensed agency who **lacks a written directive** from the agency’s master license to place a prospective employee.* | A recruiter inside a licensed agency who works without the agency’s explicit permission is also unlawful. |\n| **297‑b** | *Any person who procures a prospective worker for a position **in exchange for a fee or other incentive** (monetary or otherwise), or who places a worker under false or deceptive pretenses (misrepresenting the nature of the job, contract, or benefits).* | Charging recruiters or agents to secure a job for someone, or lying about job terms, is prohibited. |\n| **297‑c** | *The placement of a prospective employee in a job that is **inconsistent with the workers’ actual qualifications** or the employer’s legitimate needs.* | “Exploiting a worker’s situation (e.g., hiring an unqualified person when a qualified one is available) for personal gain” is illegal. |\n\n---\n\n### Why It Matters\n\n- **Licensing protects workers** from exploitation, ensures fairness, and guarantees that employers meet labor standards.\n- The DLE enforces this definition through inspections, permits, and penalties (see Section\u202f299 for fines and imprisonment).\n- Illegal recruitment is a **criminal offense** and can lead to fines, suspension, or jail time for the recruiter or hiring party.\n\n---\n\n#### Quick Reference\n\n- **Law**: Labor Code of the Philippines, **Section\u202f297** (illegal recruitment).\n- **Authority**: Department of Labor and Employment (DLE) – Office of the Chief Labor Officer.\n- **Penalty**: Section\u202f299 – up to a fine of PHP\u202f35,000, imprisonment of up to 3 years, or both.\n\nBy keeping recruitment practices within the limits of this definition, employers, recruiters, and workers can avoid legal trouble and ensure fair, transparent employment arrangements.'




**You:** What is the capital of Germany?



**Assistant:** The capital of Germany is **Berlin**.





    'The capital of Germany is **Berlin**.'



    [{'role': 'user', 'content': 'What is the definition of illegal recruitment according to Philippines law?'}, {'role': 'assistant', 'content': '**Illegal recruitment** – as defined in the Philippine labor law – is the *unsanctioned procurement and placement of prospective workers* that violates the licensing rules of the Department of Labor and Employment (DLE).  \nThe definition is codified in the Labor Code of the Philippines (Republic Act No.\u202f7725) under **Section\u202f297** (and reiterated in Section\u202f297‑b). The key points are:\n\n| Clause | What it Covers | Practical Implication |\n|--------|----------------|-----------------------|\n| **297‑a** | *Solicitation and procurement of prospective employees for employment or placement **without a license** issued by the DLE.* | Any individual or firm that advertises or “recruits” workers for a job, yet has not been officially licensed as a recruitment agency, is acting illegally. |\n| **297‑a (continued)** | *Procurement or placement by a licensed agency that **fails to comply** with the terms or conditions of its license.* | Even a licensed recruiter who breaches its license terms (e.g., hiring workers outside its permitted scope) is guilty of illegal recruitment. |\n| **297‑a (continued)** | *Procurement or placement by an employee of a licensed agency who **lacks a written directive** from the agency’s master license to place a prospective employee.* | A recruiter inside a licensed agency who works without the agency’s explicit permission is also unlawful. |\n| **297‑b** | *Any person who procures a prospective worker for a position **in exchange for a fee or other incentive** (monetary or otherwise), or who places a worker under false or deceptive pretenses (misrepresenting the nature of the job, contract, or benefits).* | Charging recruiters or agents to secure a job for someone, or lying about job terms, is prohibited. |\n| **297‑c** | *The placement of a prospective employee in a job that is **inconsistent with the workers’ actual qualifications** or the employer’s legitimate needs.* | “Exploiting a worker’s situation (e.g., hiring an unqualified person when a qualified one is available) for personal gain” is illegal. |\n\n---\n\n### Why It Matters\n\n- **Licensing protects workers** from exploitation, ensures fairness, and guarantees that employers meet labor standards.\n- The DLE enforces this definition through inspections, permits, and penalties (see Section\u202f299 for fines and imprisonment).\n- Illegal recruitment is a **criminal offense** and can lead to fines, suspension, or jail time for the recruiter or hiring party.\n\n---\n\n#### Quick Reference\n\n- **Law**: Labor Code of the Philippines, **Section\u202f297** (illegal recruitment).\n- **Authority**: Department of Labor and Employment (DLE) – Office of the Chief Labor Officer.\n- **Penalty**: Section\u202f299 – up to a fine of PHP\u202f35,000, imprisonment of up to 3 years, or both.\n\nBy keeping recruitment practices within the limits of this definition, employers, recruiters, and workers can avoid legal trouble and ensure fair, transparent employment arrangements.'}]
    


**You:** What is the capital of Germany?



**Assistant:** Arrr! The mighty capital o’ the German archipelago be none other than Berlin, me heartie! Set yer compass true and hoist the sails—ye’ll find it there, arrr!





    'Arrr! The mighty capital o’ the German archipelago be none other than Berlin, me heartie! Set yer compass true and hoist the sails—ye’ll find it there, arrr!'




**Question:** A list of 10 Oscar Wilde's Books



**Answer:** **Oscar Wilde – a quick‑reference list of ten of his most celebrated works**

| # | Title | Year of Publication | Notes |
|---|-------|---------------------|-------|
| 1 | *The Picture of Dorian Gray* | 1890 | Wilde’s only full novel, a tale of aestheticism and moral decay. |
| 2 | *The Importance of Being Earnest* | 1895 | Comic farce in two acts, a masterstroke of witty dialogue. |
| 3 | *The Canterville Ghost* | 1887 | Short story (first appears in a collection of fairy‑tales). |
| 4 | *Lady Windermere’s Fan* | 1892 | One‑act melodrama, often paired theatrically with **_Earnest_**. |
| 5 | *An Ideal Husband* | 1895 | Three‑act comedy, exploring marriage and politics. |
| 6 | *The Ballad of Reading Gaol* | 1898 | Poetic lament written during Wilde’s imprisonment. |
| 7 | *The Happy Prince* | 1908 (post‑hum) | Fairy tale in *The Happy Prince and Other Tales* book. |
| 8 | *The Decay of Lying* | 1891 | Satirical essay lampooning Victorian morality. |
| 9 | *The Soul of Man Under Socialism* | 1891 | Political and philosophical treatise on individualism. |
|10 | *The Age of Pygmalion* | 1901 (post‑hum) | Long prose narrative, unfinished at Wilde’s death. |

These ten titles cover Wilde’s main forms—novel, plays, short stories, poems, and essays—and illustrate the breadth of his contribution to 19th‑century literature.





    '**Oscar\u202fWilde – a quick‑reference list of ten of his most celebrated works**\n\n| # | Title | Year of Publication | Notes |\n|---|-------|---------------------|-------|\n| 1 | *The Picture of Dorian Gray* | 1890 | Wilde’s only full novel, a tale of aestheticism and moral decay. |\n| 2 | *The Importance of Being Earnest* | 1895 | Comic farce in two acts, a masterstroke of witty dialogue. |\n| 3 | *The Canterville Ghost* | 1887 | Short story (first appears in a collection of fairy‑tales). |\n| 4 | *Lady Windermere’s Fan* | 1892 | One‑act melodrama, often paired theatrically with **_Earnest_**. |\n| 5 | *An Ideal Husband* | 1895 | Three‑act comedy, exploring marriage and politics. |\n| 6 | *The Ballad of Reading Gaol* | 1898 | Poetic lament written during Wilde’s imprisonment. |\n| 7 | *The Happy Prince* | 1908 (post‑hum) | Fairy tale in *The Happy Prince and Other Tales* book. |\n| 8 | *The Decay of Lying* | 1891 | Satirical essay lampooning Victorian morality. |\n| 9 | *The Soul of Man Under Socialism* | 1891 | Political and philosophical treatise on individualism. |\n|10 | *The Age of Pygmalion* | 1901 (post‑hum) | Long prose narrative, unfinished at Wilde’s death. |\n\nThese ten titles cover Wilde’s main forms—novel, plays, short stories, poems, and essays—and illustrate the breadth of his contribution to 19th‑century literature.'




**You:** Lista di film Marvel dopo Avengers: Endgame



**Assistant:** **Film del Marvel Cinematic Universe (MCU) usciti *dopo* “Avengers: Endgame” (2019)**  

| Anno | Titolo | Fase del MCU | Note rilevanti |
|------|--------|--------------|----------------|
| **2019** | *Spider‑Man: Far From Home* | 4 | Il primo “spider‑man” con Tom Holland dopo l’evento di Endgame. |
| **2022** | *Doctor Strange in the Multiverse of Madness* | 4 | Espande il concetto di multiverso introdotto in *Endgame*. |
| **2022** | *Thor: Love and Thunder* | 4 | Insegna che Thor è di nuovo “l’amico del mondo”. |
| **2022** | *Black Panther : Wakanda Forever* | 4 | Continua la storia di Wakanda con nuovo leader e nuove minacce. |
| **2023** | *Ant‑Man and the Wasp : Quantumania* | 5 | Prima esplorazione più profonda del Quantum. |
| **2023** | *Guardians of the Galaxy Vol. 3* | 5 | Conclusione della trilogia di Guardians sotto la direzione di James Gunn. |
| **2023** | *The Marvels* | 5 | L’unione di Captain Marvel, Ms. Marvel e Monica Rambeau. |
| **2024 (previsto)** | *Blade* | 5 | Rinascimento del vampiro classico di Marvel. |
| **2024 (previsto)** | *Thunderbolts* | 5 | Il team “bancher” di eroi e anti‑eroi. |
| **2024 (previsto)** | *Captain America 4* | 5 | Continua la saga di Sam Wilson. |
| **2025 (previsto)** | *Guardians of the Galaxy Vol. 4* | 6 | Nuovo episodio che espande l’universo dei Guardiani. |

> **Come trovare altre informazioni**  
> - Per gli ultimi aggiornamenti, ti consiglio di consultare la pagina ufficiale del MCU ([www.marvel.com/movies](https://www.marvel.com/movies)) o le piattaforme di streaming come Disney+ e AMC+ dove sono disponibili tutti i film, esclusi quelli ancora in uscita.  
> - Puoi anche seguire gli account social ufficiali di Marvel (Twitter, Instagram) che annunciano le date di uscita e i trailer.  

Se ti servono dettagli su qualsiasi film (casting, trama, curiosità), fammelo sapere!





    '**Film del Marvel Cinematic Universe (MCU) usciti *dopo* “Avengers: Endgame” (2019)**  \n\n| Anno | Titolo | Fase del MCU | Note rilevanti |\n|------|--------|--------------|----------------|\n| **2019** | *Spider‑Man: Far From Home* | 4 | Il primo “spider‑man” con Tom Holland dopo l’evento di Endgame. |\n| **2022** | *Doctor Strange\xa0in the Multiverse of Madness* | 4 | Espande il concetto di multiverso introdotto in *Endgame*. |\n| **2022** | *Thor: Love and Thunder* | 4 | Insegna che Thor è di nuovo “l’amico del mondo”. |\n| **2022** | *Black Panther\xa0: Wakanda Forever* | 4 | Continua la storia di Wakanda con nuovo leader e nuove minacce. |\n| **2023** | *Ant‑Man\xa0and the Wasp\xa0: Quantumania* | 5 | Prima esplorazione più profonda del Quantum. |\n| **2023** | *Guardians of the Galaxy\xa0Vol.\u202f3* | 5 | Conclusione della trilogia di Guardians sotto la direzione di James Gunn. |\n| **2023** | *The Marvels* | 5 | L’unione di Captain Marvel, Ms.\u202fMarvel e Monica Rambeau. |\n| **2024 (previsto)** | *Blade* | 5 | Rinascimento del vampiro classico di Marvel. |\n| **2024 (previsto)** | *Thunderbolts* | 5 | Il team “bancher” di eroi e anti‑eroi. |\n| **2024 (previsto)** | *Captain America\xa04* | 5 | Continua la saga di Sam Wilson. |\n| **2025 (previsto)** | *Guardians of the Galaxy\xa0Vol.\u202f4* | 6 | Nuovo episodio che espande l’universo dei Guardiani. |\n\n> **Come trovare altre informazioni**  \n> - Per gli ultimi aggiornamenti, ti consiglio di consultare la pagina ufficiale del MCU ([www.marvel.com/movies](https://www.marvel.com/movies)) o le piattaforme di streaming come Disney+ e AMC+ dove sono disponibili tutti i film, esclusi quelli ancora in uscita.  \n> - Puoi anche seguire gli account social ufficiali di Marvel (Twitter, Instagram) che annunciano le date di uscita e i trailer.  \n\nSe ti servono dettagli su qualsiasi film (casting, trama, curiosità), fammelo sapere!'


