class Statistics:
    def __init__(self, dataset):
        self.dataset = dataset
        # Define a ordem lógica para colunas ordinais (baixa < media < alta)
        self._order_map = {"baixa": 1, "media": 2, "alta": 3}

    def _get_sorted_data(self, column):
        # Função auxiliar para ordenar respeitando a lógica de prioridade.
        raw_data = self.dataset[column]
        # Se a coluna for 'priority', ordena pelo mapa de pesos
        if column == "priority":
            return sorted(raw_data, key=lambda x: self._order_map.get(x, 0))
        return sorted(raw_data)

    def mean(self, column):
        data = self.dataset[column]
        return sum(data) / len(data)

    def median(self, column):
        # - Mediana é o valor central de uma distribuição.
        # Usamos nossa função de ordenação customizada
        data = self._get_sorted_data(column)
        n = len(data)
        mid = n // 2

        if n % 2 != 0:
            return data[mid]
        else:
            # Se forem strings (como 'media'), não podemos somar.
            # Em estatística, para medianas de listas pares de strings,
            # escolhe-se o elemento à esquerda ou o valor central.
            if isinstance(data[mid], str):
                return data[mid]
            return (data[mid - 1] + data[mid]) / 2

    def mode(self, column):
        # - Moda é o valor mais frequente em uma distribuição.
        data = self.dataset[column]
        counts = {}
        for item in data:
            counts[item] = counts.get(item, 0) + 1

        max_f = max(counts.values())
        # O teste espera uma lista. Retornamos apenas as chaves com valor máximo.
        return [k for k, v in counts.items() if v == max_f]

    def variance(self, column):
        # - Variância indica o quão longe uma distribuição se encontra da media.
        data = self.dataset[column]
        m = self.mean(column)
        return sum((x - m) ** 2 for x in data) / len(data)

    def stdev(self, column):
        return self.variance(column) ** 0.5

    def covariance(self, column_a, column_b):
        # - Covariância é uma medida de correlação entre duas variáveis.
        data_x, data_y = self.dataset[column_a], self.dataset[column_b]
        m_x, m_y = self.mean(column_a), self.mean(column_b)
        n = len(data_x)
        return sum((data_x[i] - m_x) * (data_y[i] - m_y) for i in range(n)) / n

    def itemset(self, column):
        return set(self.dataset[column])

    def absolute_frequency(self, column):
        # - Frequência absoluta é a contagem direta de ocorrências.
        data = self.dataset[column]
        freq = {}
        for item in data:
            freq[item] = freq.get(item, 0) + 1
        return freq

    def relative_frequency(self, column):
        # - Proporção de uma categoria em relação ao total.
        abs_f = self.absolute_frequency(column)
        n = len(self.dataset[column])
        return {k: v / n for k, v in abs_f.items()}

    def cumulative_frequency(self, column, frequency_method="absolute"):
        # - Frequência acumulada (soma das frequências anteriores).
        if frequency_method == "absolute":
            f_dict = self.absolute_frequency(column)
        else:
            f_dict = self.relative_frequency(column)

        # Ordenamos as chaves pelo nosso mapa de prioridade
        if column == "priority":
            sorted_keys = sorted(f_dict.keys(), key=lambda x: self._order_map.get(x, 0))
        else:
            sorted_keys = sorted(f_dict.keys())

        cum = {}
        total = 0
        for k in sorted_keys:
            total += f_dict[k]
            cum[k] = total
        return cum

    def conditional_probability(self, column, value1, value2):
        # - Probabilidade de A ocorrer dado que B ocorreu.
        data = self.dataset[column]
        count_b = 0
        count_ba = 0
        for i in range(len(data) - 1):
            if data[i] == value2:  # Evento B (condição)
                count_b += 1
                if data[i + 1] == value1:  # Evento A (consequente)
                    count_ba += 1
        return count_ba / count_b if count_b > 0 else 0.0

    def quartiles(self, column):
        # - Quartis dividem uma distribuição em quatro partes iguais.
        data = sorted(self.dataset[column])
        n = len(data)

        # Método de Interpolação Linear
        def get_percentile(sorted_list, p):
            idx = p * (n - 1)
            lower = int(idx)
            upper = lower + 1
            weight = idx - lower
            if upper >= n:
                return float(sorted_list[lower])
            return sorted_list[lower] * (1 - weight) + sorted_list[upper] * weight

        return {
            "Q1": get_percentile(data, 0.25),
            "Q2": get_percentile(data, 0.50),
            "Q3": get_percentile(data, 0.75),
        }

    def histogram(self, column, bins):
        # - Divisão dos dados em intervalos (buckets) de igual tamanho.
        data = self.dataset[column]
        v_min, v_max = min(data), max(data)
        width = (v_max - v_min) / bins

        hist = {}
        intervals = []
        for i in range(bins):
            start = v_min + i * width
            end = start + width
            if i == bins - 1:
                end = v_max + 0.0001
            intervals.append((start, end))
            hist[(start, end)] = 0

        for v in data:
            for start, end in intervals:
                if start <= v < end:
                    hist[(start, end)] += 1
                    break
        return hist
